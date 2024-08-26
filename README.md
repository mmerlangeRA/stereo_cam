# Stereo cam

## Objectif

A partir de 2 images prises par 2 caméras dont la position et rotation relative est connue :

* estimer par triangulation la position d'un point commun entre les 2 images
* estimer la largeur d'une route à partir d'1 ou 2 images

## Local installation and development

### Virtual env

La première fois :

* création de l'environnement virtuel

```bash
pyenv install 3.11
pyenv local 3.11
python3.11 -m venv venv
source venv/bin/activate
python bootstrap.py
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

### Autocalibration EAC

```bash
python triangulate.py auto_calibrate --imgLeft_name "Photos/P1/D_P1_CAM_G_0_EAC.png" --imgRight_name "Photos/P1/D_P1_CAM_D_0_EAC.png" --initial_params "[0, 0, 0, 1.12, 0, 0]" --bnds "[[-0.17, 0.17], [-0.17, 0.17], [-0.17, 0.17], [1.11, 1.13], [-0.12001, 0.12001], [-0.12001, 0.12001]]" --inlier_threshold 0.01 
```

### Triangulation EAC

R et t sont les 3 premiers/derniers paramètres retournés par auto_calibrate

```bash
python triangulate.py triangulatePoints --keypoints_cam1 "[100.0, 200.0]" --keypoints_cam2 "[150.0, 250.0]" --image_width 5376 --image_height 2388 --R "[0.0, 0.0, 0.0]" --t "[1.12, 0.0, 0.0]"

```

### Test de triangulation EAC

Test simple pour régénérer <https://docs.google.com/spreadsheets/d/1hidqo7HglxUd2cEmL3stfshqK71vDYl6QxuA9wgwzOo/edit?usp=sharing>:

Il est supposé que toutes les photos sont dans le dossier Photos, à la racine.

J'ai utilisé une copie de <https://drive.google.com/drive/folders/1DFFxjpu4VPXhJ9-PGA0izL2VoJxEwOxL?usp=sharing>

```bash
python triangulate_test.py

```

### Depth estimation

```bash
python stereo_depth.py --restore_ckpt pretrained_models\middlebury_finetune.pth --valid_iters 180 --max_disp 768 --left_img C:\Users\mmerl\projects\stereo_cam\static\photos\13_rectified_left.jpg --right_img C:\Users\mmerl\projects\stereo_cam\static\photos\13_rectified_right.jpg  --output_directory output
```

### Road segmentation

Stereo, standard

```bash
python scripts/road_detection_stereo.py --img_left_path C:\Users\mmerl\projects\stereo_cam\undistorted_CUBE\11_rectified_left.jpg --img_right_path C:\Users\mmerl\projects\stereo_cam\undistorted_CUBE\11_rectified_right.jpg --calibration_path C:\Users\mmerl\projects\stereo_cam\calibration\calibrator_matrix.json
```

python scripts/road_detection_stereo.py --img_left_path C:\Users\mmerl\projects\stereo_cam\static\gauche.png --img_right_path C:\Users\mmerl\projects\stereo_cam\static\droite.png --calibration_path C:\Users\mmerl\projects\stereo_cam\calibration\calibrator_matrix.json
"C:\Users\mmerl\projects\stereo_cam\static\gauche.png"

EAC

```bash
python scripts/road_detection_eac.py --img_path C:\Users\mmerl\projects\stereo_cam\Photos\P5\D_P5_CAM_G_0_EAC.png 
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

## Streamlit

```bash
streamlit run simple_app.py
```

Hack Windows Michaël

```bash
C:\Users\mmerl\anaconda3\envs\logiroadhitnet\python.exe -m streamlit run simple_app.py
```

## Bibliothèques tierces

* segmentation : <https://github.com/XuJiacong/PIDNet>

Pour que cela fonctionne, il faut ajouter dans src/pretrained_models le dossier cityscapes disponible ici : <https://drive.google.com/drive/folders/1xDwOiH-Z0cOK_F6lvnykVyjry1s2njRS?usp=sharing>

## Tests unitaires (nouveau, en cours)

Dans le dossiers tests

```bash
python -m unittest discover -s tests
```

## Docker installation

to be done
