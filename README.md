# ml-product-hw

Домашние задания по Управлению ML продуктом 1

## Установка и запуск

Нужно установить git, make и [uv](https://docs.astral.sh/uv/). Python 3.12 uv скачает сам.

```bash
git clone https://github.com/Wrongass/ml-product-hw.git
cd ml-product-hw
make setup
```

Команды:

- `make setup` - создать окружение (`uv sync`) и поставить git-хуки pre-commit
- `make check` - проверить код через ruff и mypy, ничего не исправляет
- `uv run python -m ml_product_hw.debug` - запустить отладочный скрипт

## Данные

Для этой части нужен еще docker с docker compose.

### Хранилище

Локальное S3 это RustFS в docker compose. Логин, пароль, адрес и имя бакета берутся из `.env`. Его не нужно создавать руками, `make setup` и `make s3-up` сами копируют `.env.example` в `.env`, если его нет.

```bash
make s3-up    # поднять хранилище, бакет cifar10 создается автоматически
make s3-down  # остановить (данные остаются в volume)
```

Посмотреть содержимое можно в веб-консоли http://localhost:9001/rustfs/console/ (логин и пароль из `.env`) или через aws cli:

```bash
source .env
docker run --rm --network ml-product-hw \
  -e AWS_ACCESS_KEY_ID=$S3_ACCESS_KEY -e AWS_SECRET_ACCESS_KEY=$S3_SECRET_KEY \
  amazon/aws-cli:2.38.0 --endpoint-url http://rustfs:9000 --region us-east-1 \
  s3 ls s3://cifar10/processed/cifar10/v1/
```

### Порядок команд

```bash
make setup
make s3-up
make ingest   # скачать архив с сайта CIFAR и положить в s3 как есть
make data     # весь конвейер обработки, результат выгружается в s3 как v1
```

`make ingest` ничего не делает, если архив уже лежит в S3 (и никогда не перезаписывает `raw/`). Сайт отдает архив медленно, у меня скачивание заняло около 40 минут.

Параметры `make data` можно поменять без правки кода:

```bash
make data SEED=7 VAL_SIZE=5000 VERSION=v2
```

По умолчанию `SEED=42`, `VAL_SIZE=5000`, `VERSION=v1`. Если такая версия уже есть в S3 (у нее есть manifest.json), `make data` сразу падает с ошибкой, нужно указать другую `VERSION`. Если прошлая выгрузка оборвалась и manifest.json нет, версию можно загрузить заново.

`make data` запускает по очереди шаги, каждый можно запустить отдельно (параметры смотреть через `--help`):

1. `uv run python -m ml_product_hw.data.download` - скачать архив из S3 в `data/raw/`
2. `uv run python -m ml_product_hw.data.convert` - распаковать pickle в png и json разметку в `data/processed/`
3. `uv run python -m ml_product_hw.data.split --val-size 5000 --seed 42` - стратифицированно выделить val из train
4. `uv run python -m ml_product_hw.data.stats` - посчитать mean и std по каналам только по train
5. `uv run python -m ml_product_hw.data.upload --version v1` - выгрузить в S3

### Что лежит в S3

```
cifar10/
    raw/cifar-10-python.tar.gz        # исходный архив без изменений
    processed/cifar10/v1/
        images/                       # 60000 png, train_00000.png ... test_09999.png
        annotations/
            train.json
            val.json
            test.json
        stats.json                    # mean и std по каналам RGB
        manifest.json                 # число картинок, sha256 файлов, seed, val_size, git коммит
```

В json разметке для каждой картинки есть `image_id`, `file_name`, `label`, `label_name` и `sha256` png файла.

### Карточка датасета

- Источник: CIFAR-10, https://www.cs.toronto.edu/~kriz/cifar.html (версия для Python)
- Размер: 60000 цветных картинок 32x32, 50000 в исходном train и 10000 в test
- Классы (10): airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck. Классы сбалансированы, по 6000 картинок на класс
- Разбиение: из исходного train стратифицированно по классам выделен val. По умолчанию train 45000 / val 5000 (по 500 на класс), seed 42. Test не меняется. Картинки из val сохраняют имена `train_...`
- Нормализация: mean и std в `stats.json` посчитаны только по train в шкале 0-1 (пиксели / 255). Для v1 mean примерно (0.491, 0.482, 0.446), std примерно (0.247, 0.243, 0.262)
- Известные проблемы:
  - очень маленькое разрешение, многие объекты трудно различить даже человеку
  - похожие классы, модели часто путают cat и dog, automobile и truck
  - небольшая часть картинок размечена неправильно

## Обучение

Обучение тоже идет в docker, все контейнеры в одной сети `ml-product-hw`.

### Инфраструктура

```bash
make infra-up    # собрать образ для обучения, поднять RustFS, PostgreSQL и MLflow
make infra-down  # остановить все (данные остаются в volumes)
```

- MLflow открывается на http://localhost:5000
- метаданные запусков хранятся в PostgreSQL, артефакты (модели, картинки) в бакете `mlflow` в том же RustFS
- бакеты `cifar10` и `mlflow` создаются автоматически
- пароли берутся из `.env`. Если `.env` остался от прошлого ДЗ, в него нужно дописать новые переменные из `.env.example`

В S3 должна быть версия данных `processed/cifar10/v1`. Если ее нет, нужно сначала выполнить `make ingest` и `make data` (см. выше).

### Запуск

```bash
make train CONFIG=configs/baseline.yaml     # один запуск в контейнере
make experiments CONFIG=configs/grid.yaml   # сетка, каждый запуск в отдельном контейнере
```

Конфиги лежат в `configs/`:

- `baseline.yaml` - один линейный слой поверх пикселей
- `simple_cnn.yaml` - сверточная сеть из 2 слоев, она же базовый конфиг для сетки
- `grid.yaml` - базовый конфиг, шаблон имени запуска и списки значений, которые перебираются

Один запуск скачивает данные из S3 в `data/train/`, если их там нет. Модель с лучшим val loss в конце проверяется на test. Все пишется в MLflow, а модель, конфиг и метрики еще и в S3 в `models/<experiment_name>/<run_name>/`.

Скрипт сетки (`ml_product_hw.train.grid`) строит все комбинации, сохраняет готовые конфиги в `tmp/grid/` и для каждой запускает отдельный контейнер через `docker run`. Сетка по умолчанию облегченная: 20000 картинок из train и 8 эпох. У меня на CPU с 16 потоками вся сетка заняла 347 секунд.

Чтобы поставить другой эксперимент, код менять не нужно: достаточно поменять значение в конфиге (например `train.lr` или `model.params.channels`) и `run_name`, потом снова запустить `make train` или `make experiments`.

## Эксперименты

Все запуски на данных `processed/cifar10/v1`, seed 42. Val и test считаются для модели с лучшим val loss.

| Запуск | Train | Эпохи (лучшая) | val loss | val accuracy | val F1 | test loss | test accuracy | test F1 |
|---|---|---|---|---|---|---|---|---|
| baseline `linear-lr0.001-bs64` | 45000 | 10 (2) | 1.9075 | 0.3690 | 0.3558 | 1.9196 | 0.3638 | 0.3505 |
| сетка `cnn-lr0.001-bs128` | 20000 | 8 (7) | 0.9607 | 0.6700 | 0.6701 | 0.9859 | 0.6667 | 0.6658 |
| сетка `cnn-lr0.001-bs64` | 20000 | 8 (5) | 0.9944 | 0.6568 | 0.6615 | 1.0370 | 0.6476 | 0.6512 |
| сетка `cnn-lr0.0003-bs64` | 20000 | 8 (7) | 1.0190 | 0.6534 | 0.6545 | 1.0332 | 0.6444 | 0.6437 |
| итог `cnn-lr0.001-bs128-full` | 45000 | 20 (6) | 0.8318 | 0.7236 | 0.7237 | 0.8650 | 0.7184 | 0.7180 |

Остальные запуски сетки: val accuracy 0.6056 (lr 0.003, bs 64), 0.6014 (lr 0.003, bs 128) и 0.6302 (lr 0.0003, bs 128).

Выбрана конфигурация lr 0.001 и batch size 128: в сетке у нее самый маленький val loss и самые высокие val accuracy и F1. Ее обучили на всем train 20 эпох (`configs/final.yaml`), лучшая эпоха 6, дальше модель переобучается и val loss растет. Итоговая CNN дает 0.7184 accuracy на test, baseline из одного линейного слоя 0.3638.

## Serving

Модель работает в Triton Inference Server на CPU, снаружи к ней ходят через HTTP-обертку на FastAPI.

### Какая модель развернута

- эксперимент `cifar10-final`, запуск `cnn-lr0.001-bs128-full` (итоговая модель из раздела Эксперименты)
- в Triton модель называется `cifar_cnn`, версия `1`
- данные `processed/cifar10/v1`, из этой же версии берется `stats.json` для нормализации

Путь к запуску, имя модели и версия данных лежат в `configs/serving.yaml`.

В Triton запущено 2 экземпляра модели. Модель маленькая и считается быстро, а процессор общий с оберткой и клиентом, поэтому 2 экземпляров хватит.

### Запуск

```bash
make s3-up        # или make infra-up, нужен S3 с моделью и данными
make export       # скачать модель и stats.json, экспорт в ONNX, собрать model_repository/
make serve-up     # поднять Triton (порты 8000, 8001, 8002) и обертку (порт 8081)
make verify       # проверить метрики через обертку
make load-test    # нагрузочный тест
make serve-down   # остановить все
```

`make export` нужно запускать до `make serve-up`: Triton читает `model_repository/` при старте. `model_repository/` и `serving_data/` (там лежит `stats.json` для обертки) создаются скриптом и в git не попадают. `config.pbtxt` хранится в `configs/triton/`.

Пример запроса:

```bash
curl -F file=@data/train/processed/cifar10/v1/images/test_00000.png http://localhost:8081/predict
```

```json
{"class_id": 5, "class_name": "dog", "probabilities": [0.0007, 0.0015, ...], "model_name": "cifar_cnn", "model_version": "1"}
```

`GET /health` отвечает 200, если Triton и модель готовы, и 503, если нет. Если отправить не картинку или пустой файл, обертка отвечает 400.

### Проверка метрик

`make verify` отправляет все 10000 картинок test через `POST /predict`, считает метрики по вероятностям из ответа и сравнивает с test из `metrics.json` запуска.

| Метрика | При обучении | В сервисе | Разница | Допуск |
|---|---|---|---|---|
| loss | 0.8650 | 0.8650 | 0.0000 | 0.01 |
| accuracy | 0.7184 | 0.7184 | 0.0000 | 0.002 |
| F1 (macro) | 0.7180 | 0.7180 | 0.0000 | 0.002 |

Допуски задаются аргументами `--loss-tol`, `--accuracy-tol` и `--f1-tol`. Препроцессинг в обертке тот же, что при обучении, поэтому разница может быть только из-за мелких отличий float32 между PyTorch и ONNX Runtime. Допуск 0.002 для accuracy и F1 это 20 картинок из 10000, для loss допуск 0.01. С неверным `stats.json` accuracy падает до 0.54, и проверка не проходит.

### Нагрузочный тест

Результаты, условия замера и выводы в [reports/load_test.md](reports/load_test.md). Сырые замеры в `reports/load_test_raw.csv`.
