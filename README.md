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
