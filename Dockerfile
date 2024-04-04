FROM python:3.10-slim
ENV DEBIAN_FRONTEND='noninteractive'

RUN apt-get update && apt install -y curl

RUN curl -sSL https://install.python-poetry.org | python

ENV PATH="${PATH}:/root/.local/bin"

RUN poetry config virtualenvs.in-project true

WORKDIR /app

COPY . .

RUN poetry install

CMD ["poetry", "run", "streamlit", "run", "./main.py", "--client.showErrorDetails=false"]