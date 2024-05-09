import os
import shutil

from utils.colors import Color

"""
@:param directory_results - Locatia(Directorul) unde dorim sa salvam rezultatele
@:param path_rezultatele - Directorul pe care dorim sa il adaugam ca si subfolder
Orice am face, rezultatele sunt MEREU SALVATE(eventual chiar si prin overwriting) in path ul din path_rezultate
DE aceea, DUPA ce toate rezultatele sunt slavate in directorul din path_rezultate, il adaugam CA subfolder
in alt loc si STERGEM acest folder
Adaugam path_rezultate ca subfolder in ResultsAll
"""


def save_results_desired_folder(directory_results: str, path_rezultate: str):
    final_path = os.path.join(directory_results, path_rezultate)

    # Check if the final directory already exists
    if not os.path.exists(final_path):
        os.makedirs(final_path)

    # Move each item from the source directory to the final directory
    for item in os.listdir(path_rezultate):
        source_path = os.path.join(path_rezultate, item)
        destination_path = os.path.join(final_path, item)

        # Move each item to the final directory
        shutil.move(source_path, destination_path)

    # Directoru (acum gol) path_reultate il stergem
    if os.path.exists(path_rezultate):
        shutil.rmtree(path_rezultate)


"""
@:param path_weights - Path ul catre folder ul results(unde CAUTAM cele mai bune weights obtinute) DIN
cadrul ultimei dati a antrenarii modelului (acestea sunt salvate intr-un folder de forma
ResultAll/results_nrEpoci_nrBatch_dimImage/train/weights/results_nano_10_16_640_best_weights.pt, in folder ul ResultAll, am vzt)
"""


def save_best_weights_obtained(path_rezultate: str):
    # ResultsAll / results_nano_1_16_1 / train / weights / best.pt; Exemplu path
    path_weights = ""
    path_weights += 'ResultsAll/'
    path_weights += f'{path_rezultate}/train/weights/best.pt'

    destination_directory = 'trained_models_best_weights'

    # Definim numele si locatia unde salvam fisierul
    path_salvare_best_weights = f'{path_rezultate}_best_weights.pt'
    path_salvare_best_weights = os.path.join(destination_directory, path_salvare_best_weights)  # New file name and path

    # Copiem fisierul
    shutil.copy(path_weights, path_salvare_best_weights)


"""
@:param path_results - Path ul catre noul folder ce UNDE dorim a salva rezultatele
@:return True Daca modelul VA fi antrenat pe aceste date, False in caz contrar
VERIFICAM daca exista deja o rulare EXACT cu astfel de date introduse. Daca exista, intrebam pe utilizator
daca doreste sa continue. Daca da, atunci stergem datele respective OBTINUTE, si salvam datele pentru
aceasta noua rulare(Posibil ca acesta sa fi schimbal alti hyper parametrii de care nu tinem cont, precum functia
de optimizare, functia pierdere, etc)!!

"""


def check_exist_run(path_rezultate: str) -> bool:
    directory_results = 'ResultsAll'

    # VEDEM daca am avut o rulare cu aceste date
    check_exist_run = os.path.join(directory_results, path_rezultate)
    if os.path.exists(check_exist_run):
        optiune = ""
        while not (optiune.__eq__("DA") or optiune.__eq__("NU")):
            optiune = input("AVETI deja o rulare cu aceste date! Sigur doriti sa continuati? Se vor suprascrie "
                            "TOATE datele deja salvate! (DA | NU): ")

        # Se razgandeste utilizatorul, NU face rularea!
        if optiune.__eq__("NU"):
            return False

        # Stergem folder ul cu rezultate si best_weight salavte, if ul de verificare e de siguranta!
        if optiune.__eq__("DA"):
            if os.path.exists(check_exist_run):
                shutil.rmtree(check_exist_run)

            best_weights = 'trained_models_best_weights'

            # Facem loop prin fisierele din director, si obtinem DOAR numele fisierelor (nume_fisier.extensie)
            for filename in os.listdir(best_weights):
                # Cnstruim tot path ul
                file_path = os.path.join(best_weights, filename)

                # Check if the filename starts with 'path_rezultate'
                if filename.startswith(path_rezultate):
                    os.remove(file_path)
                    # print(f"Deleted: {file_path}")
            return True

    # DACA nu exista deja rezultatele, RULAREA se face automat!
    return True

"""
@:return Un dictionar cu clasele si numerele asociate lor (SIMILAR cu format ul din config.yaml), DAR aici
vrem ca NUMELOR de clase sa le asociem indecsi
Aceasta functie este folostia, spre exemplu, in menu.py in plot_boxes(deci cand vrem sa afisam clase)
"""
def construct_classes_dictionary() -> dict:
    classes ={
    "Icon": 0,
    "TextLabel": 1,
    "MenuItem": 2,
    "Row": 3,
    "Button": 4,
    "SubmenuItem": 5,
    "NavigationItem": 6,
    "DropdownItem": 7,
    "InputField": 8,
    "SectionTitle": 9,
    "TitleBar": 10,
    "Menu": 11,
    "WorkingArea": 12,
    "VerticalMenu": 13,
    "NavigationMenu": 14,
    "TableHeader": 15
    }
    return classes

"""
@:return Un dictionar cu culorile asociate claselor
Am vazut ca fiecarei clase ii corespunde UN UNIC index, si vrem ca fiecarui index sa ii corespunda o unica CULOARE!
Practic definim niste bijectii intre Nume clase <-> indecsi <-> culori
Tinem cont ca culorile sunt ENUMS!!! (Deci avme si Nume si Valoarea!)
"""
def construct_colors_dictionary() -> dict:
    colors_associated = dict()
    for index, color in enumerate(Color):
        colors_associated[index] = color.value
    return colors_associated
