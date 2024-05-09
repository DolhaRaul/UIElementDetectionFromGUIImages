import os
import shutil

import cv2
from ultralytics import YOLO

from utils.utils import check_exist_run
from utils.utils import construct_classes_dictionary
from utils.utils import construct_colors_dictionary
from utils.utils import save_best_weights_obtained
from utils.utils import save_results_desired_folder

# Set device to CPU


"""
Meniul principal
"""


def show_menu():
    mode = ""
    while not (mode.__eq__("train") or mode.__eq__("predict")):
        mode = input("Introduceti optiunea modelului aici (train | predict): ")
        if mode.__eq__("train"):
            model = train_mode()
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
    while not (transfer_learning.__eq__("DA") or transfer_learning.__eq__("NU")):
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


"""
Meniul pentru model in faza de predict
"""


def show_predict():
    classes = ['Icon', 'TextLabel', 'MenuItem', 'Row', 'Button', 'SubmenuItem',
               'NavigationItem', 'DropdownItem', 'InputField', 'SectionTitle',
               'TitleBar', 'Menu', 'WorkingArea', 'VerticalMenu', 'NavigationMenu',
               'TableHeader']
    path_30epochs_trained_model = 'trained_models_best_weights/results_nano_30_16_640_best_weights.pt'
    path_to_model = input("Introduceti modelul de folosit aici (default e results_nano_30_16_640_best_weights.pt ): ")
    path_to_model = path_30epochs_trained_model if path_to_model.__eq__("") else path_to_model

    model = YOLO(path_to_model)

    path_to_image = input("Introduceti imaginea de analizat aici (default e data/images/Image1.png): ")
    path_to_image = "data/images/Image1.png" if path_to_image.__eq__("") else path_to_image

    results = model(path_to_image)

    classes_right_specified = False
    elements = []
    while classes_right_specified is not True:
        elements_to_predict = input(
            "Introduceti elementele ce doresc a fi identificate, separate printr-un spatiu: " + classes.__repr__() +
            "\n" + "Default sunt identificate toate clasele: ")
        elements = classes if elements_to_predict.__eq__("") else elements_to_predict.split()

        classes_right_specified = verify_classes(classes=classes, certain_classes=elements)

    font_thickness = input("Alegeti grosimea label urilor ce se identifica, "
                               "numar intreg (default e 1, grosime normala): ")
    font_thickness = 1 if font_thickness.__eq__("") else int(font_thickness)
    plot_prediction(model=model, results=results, input="data/images/Image1.png",
                    classes=elements, font_thickness=font_thickness)


"""
@:param classes - Lista de clase (CELE 16) din care poate alege utilizatorul
@:param certain_classes - Clasele alese de utilizator
@:return True daca utilizatorul NU a produs typo-uri (A introdus toate clasele corect) sau False daca utilizatorul
doreste sa caute clase ce NU sunt printre cele ce doresc a fi identificate
"""


def verify_classes(classes: list[str], certain_classes: list[str]) -> bool:
    for chosen_class in certain_classes:
        if not classes.__contains__(chosen_class):
            print(f'{chosen_class} nu este printre clasele posibile!')
            return False
    return True


"""
@:param model - Modelul YOLO ce il folosim pentru predictie
@:param results - Obiectul results ce l am obtinut in urma predictiei pe o anumita imagine
@:param input - Imaginea ce am folosit o pentru predictie
@:param classes - Clasele de dorim sa le identificam in imaginea repsectiva
@:param font_thickness - Dimensiunea fontului
@:return PE imaginea ce am folosit o drept predictie, DESENAM toate bounding box urile
ce le-am obtinut, plus iconita + scor de incredere de asupra!
"""


def plot_prediction(model: YOLO, results: list, input: str, classes: list[str], font_thickness: int):
    # Setari font implicite!
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7

    # fiecarei clase ii asociem un index
    classes_with_indexs = construct_classes_dictionary()

    # Reciproc, fiecarui index ii asociem o clasa!
    indexs_with_classes = {value: key for key, value in classes_with_indexs.items()}

    associated_colors = construct_colors_dictionary()

    img_representation = cv2.imread(input)

    class_counter = {}

    for result in results:
        boxes = result.boxes.cpu().numpy()
        for box in boxes:
            class_name = indexs_with_classes[box.cls[0]]
            # Este o clasa ce TREBUIE sa o identificam!
            if class_name in classes:
                if class_name not in class_counter:
                    class_counter[class_name] = 1
                else:
                    class_counter[class_name] += 1
                class_with_index = classes_with_indexs[class_name]
                color_for_class = associated_colors[class_with_index]
                formatted_confidence_score = "{:.1f}".format(box.conf[0])
                label_with_score = class_name + formatted_confidence_score
                text_size = cv2.getTextSize(label_with_score, font, font_scale, font_thickness)[0]
                text_width, text_height = text_size

                xyxy = box.xyxy
                # Aparent e 2 dimensional(fiecare element E UN ARRAY cu un singur element, VREM sa elim o dim)
                xyxy = xyxy.flatten()
                cv2.rectangle(img_representation, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])),
                              color_for_class, 2)
                text_x = int(xyxy[0] + (xyxy[2] - xyxy[0]) / 2 - text_width / 2)  # Centram orizontal
                text_y = int(xyxy[1]) - 3  # Punem textl putin mai sus

                cv2.putText(img_representation, label_with_score, (text_x, text_y), font, font_scale, color_for_class,
                            font_thickness)

                print(f"{class_name} : {class_counter[class_name]} a fost gasita!")

    cv2.imshow("Prediction", img_representation)
    cv2.waitKey(0)

    #Extragem numele fisierului din input path
    input_filename = os.path.basename(input)

    # NU PUTEM ADAUGA NUME DE CLASE IN FILENAME!!! Desi este foarte specific, daca VREM sa identificam MULTE CLASE(de)
    # (exemplu toate) ATUNCI numeler fisierului ESTE PREA LUNG, si nu se poate crea pentru ca WINDOWS ARE LIMITA
    # la AbsolutePath!!! Astfel, in loc de numele claselor vom pune INDECSII unici asociati! (se vede si in config.yaml)

    classes_predicted_by_index = []
    for label in classes:
        associated_index = classes_with_indexs[label]
        classes_predicted_by_index.append(str(associated_index))

    classes_predicted = '_'.join(classes_predicted_by_index)

    # Construim path de baza de unde salvam si imaginea si object counter(cele 2 fisiere)
    path_prediction_saved = 'predictions/prediction_for_' + input_filename + '_for_' + classes_predicted

    path_prediction_image_saved = path_prediction_saved + ".jpg"
    cv2.imwrite(path_prediction_image_saved, img_representation)

    path_prediction_image_counter_saved = path_prediction_saved + ".txt"
    with open(path_prediction_image_counter_saved, 'w') as f:
        for label in class_counter:
            f.write(f"{label} : {class_counter[label]}\n")

