import ast
import numpy as np
import pandas as pd
import gensim
import scipy
import sys
import torch
import torch.nn.functional as F

from pytorch_transformers import BertTokenizer, BertModel, BertForMaskedLM, AdamW
from torch.utils.data import Dataset, DataLoader
from tensorboardX import SummaryWriter

np.random.seed = 1

class DefinitionsDataset(Dataset): # train_maxlen = 103; set_to_128
    
    
    def __init__(self, path_to_csv, tokenizer, max_len=128):
        self.data = pd.read_csv(path_to_csv)
        self.data['input'] = self.data['definition'].apply(lambda x: torch.tensor(tokenizer.convert_tokens_to_ids(tokenizer.tokenize('[CLS] [MASK] - ' + x + ' [SEP]'))))
        self.data['embedding'] = self.data['embedding'].apply(lambda x: ast.literal_eval(x))
        self.data['label'] = self.data['embedding'].apply(lambda x: torch.tensor(x))
        self.data['input'] = self.data['input'].apply(lambda x: F.pad(input=x, pad=(0, max_len-(x.shape[0])), mode='constant', value=0))
        self.data['attention'] = self.data['input'].apply(lambda x: torch.tensor([float(el>0) for el in x]))
    
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        return {'input': self.data.iloc[index]['input'], 'attention_mask':self.data.iloc[index]['attention'], 'label': self.data.iloc[index]['label']}


class BertToW2v(torch.nn.Module):
    def __init__(self, bert_model_name, lin_shape_in, lin_shape_out, emb_layer): # -, 768, 100, 6
        super(BertToW2v, self).__init__()
        self.emb_layer = emb_layer
        self.bert_model = BertModel.from_pretrained(bert_model_name, output_hidden_states=True)
        #self.bert_model.eval()
        self.linear_model = torch.nn.Linear(lin_shape_in, lin_shape_out, bias=True) # bias?
        torch.nn.init.uniform_(self.linear_model.weight, -0.1, 0.1)
        
    def forward(self, input_sentence, mask): # ожидаем уже токенизированное предложение
        _, _, encoded_layers = self.bert_model(input_sentence, attention_mask=mask)
        bert_output = encoded_layers[self.emb_layer][:,1]
        linear_output = self.linear_model(bert_output)
        return linear_output


if (len(sys.argv) != 4):
    print('Wrong format: use "./model_training_batchify_clean.py PATH_TO_TRAIN PATH_TO_VALID MODEL_NAME"')
    sys.exit(0)
else:
    path_to_train = sys.argv[1]
    path_to_valid = sys.argv[2]
    model_name = sys.argv[3]

batch_size = 32
train = DefinitionsDataset(path_to_train, tokenizer)
valid = DefinitionsDataset(path_to_valid, tokenizer)

train_sample = train.data[['word', 'definition', 'input', 'attention', 'embedding']].sample(10, random_state=1) # для ускорения; 30 считаются около 80 минут
valid_sample = valid.data[['word', 'definition', 'input', 'attention', 'embedding']].sample(10, random_state=1)

bw2v = BertToW2v('bert-base-multilingual-cased', lin_shape_in=768, lin_shape_out=500, emb_layer=6) # !
bw2v.to('cuda')

train_dl = DataLoader(train, batch_size=batch_size, shuffle=True)
valid_dl = DataLoader(valid, batch_size=batch_size)

optimizer = AdamW(bw2v.parameters())
loss_function = torch.nn.MSELoss()

max_epochs = 20

for epoch in range(max_epochs):
    bw2v.train()
    train_loss = 0.0
    for dct in train_dl:
        inputs = dct['input']
        masks = dct['attention_mask']
        labels = dct['label']
        inputs = inputs.to('cuda')
        masks = masks.to('cuda')
        labels = labels.to('cuda')
        optimizer.zero_grad()
        
        output = bw2v(inputs, masks)
        loss = loss_function(output, labels)
        
        train_loss += loss.item() * inputs.size(0)
        
        loss.backward()
        optimizer.step()
        
    print('TRAINING_LOSS: ', end='')
    print(train_loss / len(train_dl.dataset), end='')
    
    bw2v.eval()
    valid_loss = 0.0
    for dct in valid_dl:
        inputs = dct['input']
        masks = dct['attention_mask']
        labels = dct['label']
        inputs = inputs.to('cuda')
        masks = masks.to('cuda')
        labels = labels.to('cuda')
        optimizer.zero_grad()
        
        with torch.no_grad():
            output = bw2v(inputs, masks)
            loss = loss_function(output, labels)
        
        valid_loss += loss.item() * inputs.size(0)
    
    print('; VALIDATION_LOSS: ', end='')
    print(valid_loss / len(valid_dl.dataset))
    
    writer.add_scalars('Loss', {'train_loss':(train_loss / len(train_dl.dataset)), 'valid_loss':(valid_loss / len(valid_dl.dataset))}, epoch+1)
    
    with torch.no_grad():
        train_sample[f'emb_epoch{epoch+1}'] = train_sample.apply(lambda row: list(bw2v(row['input'].unsqueeze(0).to('cuda'), row['attention'].unsqueeze(0).to('cuda')).cpu().numpy()[0]), axis=1)
        valid_sample[f'emb_epoch{epoch+1}'] = valid_sample.apply(lambda row: list(bw2v(row['input'].unsqueeze(0).to('cuda'), row['attention'].unsqueeze(0).to('cuda')).cpu().numpy()[0]), axis=1)

torch.save(bw2v.state_dict(), f'models/{model_name}.mdl')
writer.close()

train_sample.to_csv('rus/bert/train_sample.csv', index=None)
valid_sample.to_csv('rus/bert/valid_sample.csv', index=None)
