import os
import csv
import shutil
from PIL import Image

classes = ['Icon', 'TextLabel', 'MenuItem', 'Row', 'Button', 'SubmenuItem',
               'NavigationItem', 'DropdownItem', 'InputField', 'SectionTitle',
               'TitleBar', 'Menu', 'WorkingArea', 'VerticalMenu', 'NavigationMenu',
               'TableHeader']

"""
@:param img_name - numele fisierului imagine din data/images
@:return Latimea si inaltimea sa
Tinem cont ca toate imaginile au extensia .png, pe care o adaugam!
"""
def determine_image_dimension(img_name: str) -> [int ,int]:
    img_name += ".png"
    path_to_images = 'data/images'
    width, height = Image.open(os.path.join(path_to_images, img_name)).size
    return [width, height]



"""
@:param xmin - Abscisa punct stanga sus
@:param ymin - Ordonata punct stanga sus
@:param xmax - Abscisa punct dreapta jos
@:param ymax - Ordonata punct dreapta jos
Pentru diversele informatii legate de DEJA determinatele border box uri pentru
diferitele imagini, avand coordonatele stanfa sus - dreapta jos pentru fiecare,
VREM sa le transformam in format acceptat de YOLO pentru antrenament, adica
in format (clasa_asociata_asociata(int), centru_x, centru_y, latime, lungime)
"""

def determine_YOLO_format(clasa: str, xmin: int, ymin: int,
                          xmax: int, ymax: int) -> [int, float, float, float, float]:
    clasa_label = classes.index(clasa)  # Vedem index ul <=> label ul clasei
    centru_x = (xmax + xmin) / 2
    centru_y = (ymax + ymin) / 2
    latime = xmax - xmin
    lungime = ymax - ymin

    return [clasa_label, centru_x, centru_y, latime, lungime]

"""
@:param img_name - Numele imaginii
@:param center_x - Abscisa centrului in pixeli
@:param center_y - Ordonata centrului in pixeli
@:param width - Latimea border box ului in pixeli
@:param heigth - Inaltimea border box ului in pixeli
@:return: APARENT, pentru ca datele sa nu fie corupte, YOLOv8 TREBUIE NEAPARAT SA LUCREZE CU DATEL NORMALIZATE!
Adica coordonateke centrului bounding box, latime si inaltimea sa trebuie aduse la valori in [0, 1], si se face astfel:
normalized_x = center_x / image_width
normalized_y = center_y / image_height
normalized_width = width / image_width
normalized_height = height / image_height
"""
def normalise_bounding_box_coordinates(img_name: str, center_x: int, center_y: int,
                                       width: int, height: int) -> [float, float, float, float]:
    image_width, image_height = determine_image_dimension(img_name=img_name)
    normalized_x = center_x / image_width
    normalized_y = center_y / image_height
    normalized_width = width / image_width
    normalized_height = height / image_height
    return [normalized_x, normalized_y, normalized_width, normalized_height]


def extract_data():
    labels_path_folder = 'data/labels'
    labels_cache_path_file = 'data/labels.cache'
    # Stergem folder ele / fisierele de mai sus SI cream altele noi!!!
    if os.path.exists(labels_path_folder):
        shutil.rmtree(labels_path_folder)
    if os.path.exists(labels_cache_path_file):
        os.remove(labels_cache_path_file)

    #Cream iar labels folder
    os.mkdir(labels_path_folder)

    # Deschidem fisierul CSV
    with open('annotations.csv', newline='') as csvfile:
        reader = csv.reader(csvfile)
        data = list(reader)

    data = data[1:]

    for row in data:
        current_file = row[0]  # Vrem sa vedem din care imagine extragem bounding box
        if not os.path.exists(f'data/labels/{current_file}.txt'):  # Vedem daca avem imagine NOUA!
            with open(f'data/labels/{current_file}.txt', 'w'):
                pass
        row_YOLO_fomat = determine_YOLO_format(row[1], int(row[2]), int(row[3]),
                                               int(row[4]), int(row[5]))
        with open(f'data/labels/{current_file}.txt', 'a') as f:
            normalised_border_box = normalise_bounding_box_coordinates(current_file, *row_YOLO_fomat[1:])
            clasa_label = str(row_YOLO_fomat[0])
            centru_x = str(normalised_border_box[0])
            centru_y = str(normalised_border_box[1])
            latime = str(normalised_border_box[2])
            lungime = str(normalised_border_box[3])
            f.write(clasa_label + ' ' + centru_x + ' ' + centru_y + ' ' + latime +
                    ' ' + lungime)
            f.write('\n')

"""
In fisierul annotations.csv avem DOAR o line in plus(Table head ul); 
RESTUL informatiilor extrase trebuie sa fie IN acelasi numar
"""
def verify_data_extracted():
    # Deschidem fisierul CSV
    with open('annotations.csv', newline='') as csvfile:
        reader = csv.reader(csvfile)
        data = list(reader)

    images_directory = 'data/labels'
    line_count = 0
    for filename in os.listdir(images_directory):
        with open(f'{images_directory}/{filename}', 'r') as file:
            lines_in_file = file.readlines()
            line_count += len(lines_in_file)

    assert verify_data_extracted() == len(data) - 1