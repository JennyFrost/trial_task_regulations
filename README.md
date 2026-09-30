# команды запуска #

## установка библиотек ##
pip install -r requirements.txt
## загрузка моделей ##
python download_models.py
## создание индексов для dense retrieval и BM25 retrieval
python build_indexes.py

## запуск предсказания ##
python main.py
