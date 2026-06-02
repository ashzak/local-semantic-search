FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY semantic_search ./semantic_search
COPY web ./web
COPY docs ./docs

RUN pip install --no-cache-dir -e .

RUN semantic-search build docs --backend hashing

EXPOSE 8000

CMD ["uvicorn", "semantic_search.app:app", "--host", "0.0.0.0", "--port", "8000"]
