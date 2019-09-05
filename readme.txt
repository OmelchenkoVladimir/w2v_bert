Инструкции по запуску:

Notebook'и запускать сверху вниз (restart and run all)

Запустить notebook "SUM_data_preparation.ipynb" (подготовка данных)
Запустить скрипт "python model_training_batchify_clean.py PATH_TO_TRAIN PATH_TO_VALID MODEL_NAME", где PATH_TO_TRAIN - путь к train'у; PATH_TO_VALID - пусть к valid'у; MODEL_NAME - имя сохраняемой модели (в /models)
Запустить notebook "SUM_evaluation_by_frequency.ipynb"
Пройти в /russe_evaluation/russe/evaluation и запустить "./evaluate_test.py DATA", где DATA - полученный набор данных.
Запустить notebook "SUM_interpretable_metrics.ipynb" (получение интерпретируемых метрик)
