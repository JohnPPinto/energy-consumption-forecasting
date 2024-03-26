FROM python:3.11.7-bookworm

WORKDIR /app

RUN apt-get update && apt-get upgrade -y
RUN pip install --no-cache -U pip
RUN pip install --no-cache poetry==1.7.1

# Changing the poetry configuration for virtual environment
RUN poetry config virtualenvs.create false

COPY ./pyproject.toml ./
COPY ./poetry.lock ./
RUN touch README.md

RUN poetry install --no-interaction --without dev --no-root -vvv

ADD ./monitoring ./monitoring
ADD ./utils.py ./

CMD ["python", "-m", "streamlit", "run", "monitoring/main.py", "--server.port", "8502"]