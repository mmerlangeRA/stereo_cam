# nocode_litellm

## Description

This backend provides APIs for Illiade pipeline
- FastAPI

## Local installation and development

### Virtual env

pyenv install 3.11
pyenv local 3.11
python3.11 -m venv venv
source venv/bin/activate

### Install dependencies

pip install -r requirements.txt

### Start server

```bash
python -m uvicorn python_server.main:app --reload --host 0.0.0.0 --port 8001

```

## Docker installation

docker-compose up -d


## Documentation

APIs are fully available at site-url/docs

### Generation

```bash
python -m python_server.scripts.extract_openapi python_server.main:app --out python_server/docs/openapi.json
```
