from ultralytics import YOLO
import os
import shutil
import torch
from utils import save_results_desired_folder
from utils import save_best_weights_obtained
from utils import check_exist_run

# Set device to CPU


"""
Meniul principal
"""
def show_menu():

    mode = input("Introduceti modelul de antrenament aici (train | val | predict): ")
    model = None
    if mode.__eq__("train"):
        model = train_mode()
    if mode.__eq__("val"):
        pass
    if mode.__eq__("predict"):
       show_predict()


"""
Meniul pentru model cand e in faza de antrenament
"""
def train_mode():
    """
    De fiecare data vrem sa stergem SET UL precedent
    de date si sa il pastram pe cel nou(folosim shutil fiindca foloder ul results)
    MEREU are date in el, deci stergerea e recursiva!!!
    """
    project = 'results'
    if os.path.exists(project):
        shutil.rmtree(project)

    transfer_learning = ""
    while not (transfer_learning.__eq__("DA") or transfer_learning.__eq__("NU")) :
        transfer_learning = input("Folositi un model YOLO pre-trained sau nu (DA | NU)?: ")

    model = None
    pre_trained_type = ""
    if transfer_learning.__eq__("DA"):
        pre_trained_type = ""
        while not (pre_trained_type.__eq__("nano") or pre_trained_type.__eq__("small")
                   or pre_trained_type.__eq__("medium")):
            pre_trained_type = input("Introduceti tipul modelului ce se doreste a fi folosit (nano | small | medium): ")
            if pre_trained_type.__eq__("nano"):
                model = YOLO("yolov8n.pt")
            elif pre_trained_type.__eq__("small"):
                model = YOLO("yolov8s.pt")
            elif pre_trained_type.__eq__("medium"):
                model = YOLO("yolov8m.pt")
    elif transfer_learning.__eq__("NU"):
        model = YOLO("yolov8n.yaml")

    date = input("Introduceti fisierul de configurare(de unde se preiau datele, default e config.yaml):")
    epochs = input("Introduceti numarul de epoci aici, sau nimic pt valoarea default (100): ")
    batch_size = input("Introduceti dimensiunea batch ului aici, sau nimic pentru valoarea default (16): ")
    image_size = input("Introduceti un intreg n ce reprezinta predimensiunea la care imaginile vor fi redimensionate "
                       " sub forma n*n, sau nimic pentru o redimensionare default (640*640): ")

    date = 'config.yaml' if date.__eq__("") else date
    epochs = 100 if epochs.__eq__("") else int(epochs)
    batch_size = 16 if epochs.__eq__("") else int(batch_size)
    image_size = 640 if image_size.__eq__("") else int(image_size)

    # VREM sa indicam in path rezultate SI daca am folosit un model pre antrenat SAU nu
    if transfer_learning.__eq__("DA"):
        path_rezultate = f'results_{pre_trained_type}_{epochs}_{batch_size}_{image_size}'
    else:
        path_rezultate = f'results_newModel_{epochs}_{batch_size}_{image_size}'
    directory_results = 'ResultsAll'

    # Verificam daca AVEM / NU AVEM deja o rulare cu ACESTE date si daca FACEM rularea!
    do_train = check_exist_run(path_rezultate=path_rezultate)

    if do_train.__eq__(True):
        model.train(data=date, epochs=epochs, batch=batch_size, imgsz=image_size,
                    project=path_rezultate, device='cpu', workers=8)

        # Salvam rezultatele in folder ul ResultsAll
        save_results_desired_folder(directory_results=directory_results, path_rezultate=path_rezultate)

        # Cele mai bune weights obtinute le salvam in trained_models_best_weights
        save_best_weights_obtained(path_rezultate)


def show_predict():
    path_to_model = input("Introduceti modelul de folosit aici (default e yolov8n.pt ): ")
    path_to_model = 'yolov8n.pt' if path_to_model.__eq__("") else path_to_model

    model = YOLO(path_to_model)





