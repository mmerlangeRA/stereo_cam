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

NB : si vous ne voulez pas de serveur et pas de segmentation, minirequirements.txt devrait suffire.
Attention, il est possible que certains liens avec le serveur doivent être cassés (to be done)

Chaque fois :

* activation de l'environnemnt virtuel

```bash
source venv/bin/activate
```

## Utilisation scripts

Autocalibration

```bash
python triangulate.py auto_calibrate --imgLeft_name "Photos/P1/D_P1_CAM_G_0_EAC.png" --imgRight_name "Photos/P1/D_P1_CAM_D_0_EAC.png" --initial_params "[0, 0, 0, 1.12, 0, 0]" --bnds "[[-0.17, 0.17], [-0.17, 0.17], [-0.17, 0.17], [1.11, 1.13], [-0.12001, 0.12001], [-0.12001, 0.12001]]" --inlier_threshold 0.01 
```

Triangulation
R et t sont les 3 premiers/derniers paramètres retournés par auto_calibrate

```bash
python triangulate.py triangulatePoints --keypoints_cam1 "[100.0, 200.0]" --keypoints_cam2 "[150.0, 250.0]" --image_width 5376 --image_height 2388 --R "[0.0, 0.0, 0.0]" --t "[1.12, 0.0, 0.0]"

```

Test simple pour régénérer <https://docs.google.com/spreadsheets/d/1hidqo7HglxUd2cEmL3stfshqK71vDYl6QxuA9wgwzOo/edit?usp=sharing>:

Il est supposé que toutes les photos sont dans le dossier Photos, à la racine.

J'ai utilisé une copie de <https://drive.google.com/drive/folders/1DFFxjpu4VPXhJ9-PGA0izL2VoJxEwOxL?usp=sharing>

```bash
python triangulate_test.py

```


## Lancement du serveur sur le port 8001

```bash
python -m uvicorn python_server.main:app --reload --host 0.0.0.0 --port 8001

```

## Documentation

APIs disponibles sur site-url/docs

### Generation des apis

```bash
python -m python_server.scripts.extract_openapi python_server.main:app --out python_server/docs/openapi.json
```

## Bibliothèques tierces

* segmentation : <https://github.com/XuJiacong/PIDNet>

Pour que cela fonctionne, il faut ajouter dans src/pretrained_models le dossier cityscapes disponible ici : <https://drive.google.com/drive/folders/1xDwOiH-Z0cOK_F6lvnykVyjry1s2njRS?usp=sharing>

## Docker installation

to be done
