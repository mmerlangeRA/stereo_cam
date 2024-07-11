# Stereo cam

## Objectif

A partir de 2 images prises par 2 caméras dont la position et rotation relative est connue :

* estimer par triangulation la position d'un point commun entre les 2 images

## Local installation and development

### Virtual env

La première fois :

* création de l'environnement virtuel

```bash
pyenv install 3.11
pyenv local 3.11
python3.11 -m venv venv
source venv/bin/activate
```

* installation des dépendances

```bash
pip install -r requirements.txt
```

Chaque fois :

* activation de l'environnemnt virtuel

```bash
source venv/bin/activate
```

## Lancement du serveur sur le port 8001

```bash
python -m uvicorn python_server.main:app --reload --host 0.0.0.0 --port 8001

```

## Bibliothèques tierces

* segmentation : <https://github.com/XuJiacong/PIDNet>

## Documentation

APIs are fully available at site-url/docs

### Generation des apis

```bash
python -m python_server.scripts.extract_openapi python_server.main:app --out python_server/docs/openapi.json
```

## Docker installation

docker-compose up -d


# Ressources :
https://github.com/TemugeB/python_stereo_camera_calibrate
file:///Users/michaelargi/Downloads/MMSP_2020_SPHERE_MAPPING_FEATURE_EXTRACTION_FROM_360_FEC.pdf
https://www.ams.giti.waseda.ac.jp/data/pdf-files/2018_IWAIT_paper_105_mengcheng.pdf

