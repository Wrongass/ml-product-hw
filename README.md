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
