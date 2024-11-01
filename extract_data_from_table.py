import os
import csv
import shutil
from PIL import Image

classes = ['Icon', 'TextLabel', 'MenuItem', 'Row', 'Button', 'SubmenuItem',
               'NavigationItem', 'DropdownItem', 'InputField', 'SectionTitle',
               'TitleBar', 'Menu', 'WorkingArea', 'VerticalMenu', 'NavigationMenu',
               'TableHeader']

"""
@:param img_name - the name of the image file from data/images
@:return Its width and height
We take into account that all images have the extension .png, which we add!
"""
def determine_image_dimension(img_name: str) -> [int ,int]:
    img_name += ".png"
    path_to_images = 'data/images'
    width, height = Image.open(os.path.join(path_to_images, img_name)).size
    return [width, height]



"""
@:param xmin - Upper left point abscissa
@:param ymin - Top left point ordinate
@:param xmax - Lower right point abscissa
@:param ymax - Lower right point ordinate
For the various information related to the ALREADY determined border boxes for
the different images, having the top right - bottom right coordinates for each one,
We WANT to convert them into a format accepted by YOLO for training, that is
in format (associate_class(int), x_center, y_center, width, length)
"""

def determine_YOLO_format(clasa: str, xmin: int, ymin: int,
                          xmax: int, ymax: int) -> [int, float, float, float, float]:
    clasa_label = classes.index(clasa)  # We see the index <=> the label of the class
    centru_x = (xmax + xmin) / 2
    centru_y = (ymax + ymin) / 2
    latime = xmax - xmin
    lungime = ymax - ymin

    return [clasa_label, centru_x, centru_y, latime, lungime]

"""
@:param img_name - Name of the image
@:param center_x - The abscissa of the center in pixels
@:param center_y - The ordinate of the center in pixels
@:param width - The width of the border box in pixels
@:param heigth - The height of the border box in pixels
@:return: APPARENT, so that the data is not corrupted, YOLOv8 MUST WORK WITH NORMALIZED DATA!
That is, the coordinates of the center of the bounding box, its width and height must be brought to values ​​in [0, 1], and it is done like this:
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
    # We delete the folders / files above AND create new ones!!!
    if os.path.exists(labels_path_folder):
        shutil.rmtree(labels_path_folder)
    if os.path.exists(labels_cache_path_file):
        os.remove(labels_cache_path_file)

    #Recreate the labels folder
    os.mkdir(labels_path_folder)

    # We open the CSV file
    with open('annotations.csv', newline='') as csvfile:
        reader = csv.reader(csvfile)
        data = list(reader)

    data = data[1:]

    for row in data:
        current_file = row[0]  # We want to see from which image we extract the bounding box
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
In the annotations.csv file we ONLY have one extra line (Table head); 
The REST of the extracted information must be IN the same number
"""
def verify_data_extracted():
    # We open the CSV file
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