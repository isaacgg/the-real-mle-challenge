FROM python:3.9

WORKDIR /the-real-mle-challenge

ENV POETRY_VERSION=1.5.1

COPY ./src ./src
COPY ./data ./data
COPY ./config.yaml ./config.yaml
COPY ./models ./models
COPY ./pyproject.toml ./pyproject.toml

# Install environment and deps
RUN python -m pip install --upgrade pip
RUN pip install "poetry==$POETRY_VERSION"
RUN poetry install --no-interaction --no-ansi --no-root

# Expose the port the app runs on
EXPOSE 8008
CMD ["poetry", "run", "python", "-m", "src.api.main", "--config", "config.yaml"]
