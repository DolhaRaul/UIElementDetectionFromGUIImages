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
The main Menu
"""


def show_menu():
    mode = ""
    while not (mode.__eq__("train") or mode.__eq__("predict")):
        mode = input("Introduce the model's option here (train | predict): ")
        if mode.__eq__("train"):
            model = train_mode()
        if mode.__eq__("predict"):
            show_predict()


"""
The menu for the model when it's in the training phase
"""


def train_mode():
    """
    Every time we want to delete the previous SET UL
    of data and keep the new one ,we use shutil because the results folder
    ALWAYS has data in it, so the deletion is recursive!!!
    """
    project = 'results'
    if os.path.exists(project):
        shutil.rmtree(project)

    transfer_learning = ""
    while not (transfer_learning.__eq__("YES") or transfer_learning.__eq__("NO")):
        transfer_learning = input("Do you use a pre-trained YOLO model or not (YES | NO)?: ")

    model = None
    pre_trained_type = ""
    if transfer_learning.__eq__("YES"):
        pre_trained_type = ""
        while not (pre_trained_type.__eq__("nano") or pre_trained_type.__eq__("small")
                   or pre_trained_type.__eq__("medium")):
            pre_trained_type = input("Enter the type of model you want to use (nano | small | medium): ")
            if pre_trained_type.__eq__("nano"):
                model = YOLO("yolov8n.pt")
            elif pre_trained_type.__eq__("small"):
                model = YOLO("yolov8s.pt")
            elif pre_trained_type.__eq__("medium"):
                model = YOLO("yolov8m.pt")
    elif transfer_learning.__eq__("NU"):
        model = YOLO("yolov8n.yaml")

    date = input("Enter the configuration file (from where the data is taken, default is config.yaml):")
    epochs = input("Enter the number of epochs here, or nothing for the default value (100): ")
    batch_size = input("Enter the batch size here, or nothing for the default value (16): ")
    image_size = input("Enter an integer representing the predimension to which the images will be resized "
                       " in the form n*n, or nothing for a default resizing (640*640): ")

    date = 'config.yaml' if date.__eq__("") else date
    epochs = 100 if epochs.__eq__("") else int(epochs)
    batch_size = 16 if epochs.__eq__("") else int(batch_size)
    image_size = 640 if image_size.__eq__("") else int(image_size)

    # We WANT to indicate in the path results AND if we used a pre-trained model OR not
    if transfer_learning.__eq__("DA"):
        path_rezultate = f'results_{pre_trained_type}_{epochs}_{batch_size}_{image_size}'
    else:
        path_rezultate = f'results_newModel_{epochs}_{batch_size}_{image_size}'
    directory_results = 'ResultsAll'

    # We check if we DO / DON'T already have a run with THESE data and if we DO the run!
    do_train = check_exist_run(path_rezultate=path_rezultate)

    if do_train.__eq__(True):
        model.train(data=date, epochs=epochs, batch=batch_size, imgsz=image_size,
                    project=path_rezultate, device='cpu', workers=8)

        # We save the results in the ResultsAll folder
        save_results_desired_folder(directory_results=directory_results, path_rezultate=path_rezultate)

        # We save the best weights obtained in trained_models_best_weights
        save_best_weights_obtained(path_rezultate)


"""
The menu for the model in the prediction phase
"""


def show_predict():
    classes = ['Icon', 'TextLabel', 'MenuItem', 'Row', 'Button', 'SubmenuItem',
               'NavigationItem', 'DropdownItem', 'InputField', 'SectionTitle',
               'TitleBar', 'Menu', 'WorkingArea', 'VerticalMenu', 'NavigationMenu',
               'TableHeader']
    path_30epochs_trained_model = 'trained_models_best_weights/results_nano_30_16_640_best_weights.pt'
    path_to_model = input("Enter the model to be used here (default is results_nano_30_16_640_best_weights.pt): ")
    path_to_model = path_30epochs_trained_model if path_to_model.__eq__("") else path_to_model

    model = YOLO(path_to_model)

    path_to_image = input("Enter the image to be analyzed here (default is data/images/Image1.png): ")
    path_to_image = "data/images/Image1.png" if path_to_image.__eq__("") else path_to_image

    results = model(path_to_image)

    classes_right_specified = False
    elements = []
    while classes_right_specified is not True:
        elements_to_predict = input(
            "Enter the elements you want to be identified, separated by a space: " + classes.__repr__() +
            "\n" + "By default all classes are identified: ")
        elements = classes if elements_to_predict.__eq__("") else elements_to_predict.split()

        classes_right_specified = verify_classes(classes=classes, certain_classes=elements)

    font_thickness = input("Choose the thickness of the labels that identify, "
                               "integer number (default is 1, normal thickness):")
    font_thickness = 1 if font_thickness.__eq__("") else int(font_thickness)
    plot_prediction(model=model, results=results, input="data/images/Image1.png",
                    classes=elements, font_thickness=font_thickness)


"""
@:param classes - List of classes (THE 16) from which the user can choose
@:param certain_classes - The classes chosen by the user
@:return True if the user did NOT produce typos (Entered all classes correctly) or False if the user
wants to look for classes that are NOT among those that want to be identified
"""


def verify_classes(classes: list[str], certain_classes: list[str]) -> bool:
    for chosen_class in certain_classes:
        if not classes.__contains__(chosen_class):
            print(f'{chosen_class} is not among the possible classes!')
            return False
    return True


"""
@:param model - The YOLO model we use for prediction
@:param results - The results object that we obtained after the prediction on a certain image
@:param input - The image we used for prediction
@:param classes - The classes we want to identify in the respective image
@:param font_thickness - Font size
@:return ON the image I used as a prediction, I DRAW all the bounding boxes
what I got, plus the icon + confidence score!
"""


def plot_prediction(model: YOLO, results: list, input: str, classes: list[str], font_thickness: int):
    # Default font settings!
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7

    # we associate an index to each class
    classes_with_indexs = construct_classes_dictionary()

    # Reciprocally, we associate a class with each index!
    indexs_with_classes = {value: key for key, value in classes_with_indexs.items()}

    associated_colors = construct_colors_dictionary()

    img_representation = cv2.imread(input)

    class_counter = {}

    for result in results:
        boxes = result.boxes.cpu().numpy()
        for box in boxes:
            class_name = indexs_with_classes[box.cls[0]]
            # It is a class that we MUST identify!
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
                # Apparently it is 2 dimensional (each element IS AN ARRAY with a single element, WE WANT to remove a dime)
                xyxy = xyxy.flatten()
                cv2.rectangle(img_representation, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])),
                              color_for_class, 2)
                text_x = int(xyxy[0] + (xyxy[2] - xyxy[0]) / 2 - text_width / 2)  # We center horizontally
                text_y = int(xyxy[1]) - 3  # WE put textl a little above

                cv2.putText(img_representation, label_with_score, (text_x, text_y), font, font_scale, color_for_class,
                            font_thickness)

                print(f"{class_name} : {class_counter[class_name]} has been found!")

    cv2.imshow("Prediction", img_representation)
    cv2.waitKey(0)

    #Extract the file name from the input path
    input_filename = os.path.basename(input)

    # WE CANNOT ADD CLASS NAMES IN THE FILENAME!!! Although it is very specific, if we WANT to identify MANY CLASS(es)
    # (example all) THEN the file name IS TOO LONG, and it cannot be created because WINDOWS HAS A LIMIT
    # at AbsolutePath!!! Thus, instead of class names we will put unique associated INDEXES!
    # (it can also be seen in config.yaml)

    classes_predicted_by_index = []
    for label in classes:
        associated_index = classes_with_indexs[label]
        classes_predicted_by_index.append(str(associated_index))

    classes_predicted = '_'.join(classes_predicted_by_index)

    # We build the base path where we save the image and object counter (the 2 files)
    path_prediction_saved = 'predictions/prediction_for_' + input_filename + '_for_' + classes_predicted

    path_prediction_image_saved = path_prediction_saved + ".jpg"
    cv2.imwrite(path_prediction_image_saved, img_representation)

    path_prediction_image_counter_saved = path_prediction_saved + ".txt"
    with open(path_prediction_image_counter_saved, 'w') as f:
        for label in class_counter:
            f.write(f"{label} : {class_counter[label]}\n")

