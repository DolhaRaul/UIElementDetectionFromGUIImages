import csv
from ultralytics import YOLO
import shutil
import os
from extract_data_from_table import extract_data
import matplotlib
import torch
from menu import show_menu
#torch.backends.cudnn.enabled = True


matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from PIL import Image

# print(torch.cuda.is_available())
# print(torch.cuda.device_count())


if __name__ == '__main__':
    torch.cuda.empty_cache()
    # Check for HOW MUCH MEMORY I have allocated on the GPU!
    # torch.cuda.set_device(0)
    # if torch.cuda.is_available():
    #     # Get the memory allocated on the default GPU
    #     print(f"GPU name: {torch.cuda.get_device_name(0)}")
    #     total_memory = torch.cuda.get_device_properties(0).total_memory
    #     allocated_memory = torch.cuda.memory_allocated()
    #     # Get the max memory capacity of the default GPU
    #     max_memory = torch.cuda.max_memory_allocated()
    #     print(f"Total memory: {total_memory / 1024 ** 2} MB")
    #     print(f"Occupied memory: {allocated_memory / 1024 ** 2} MB")
    #     print(f"Max memory capacity: {max_memory / 1024 ** 2} MB")
    creare_input = input("If the correct labels folder is not already created with bounding boxes"
                         " for each specific image, create now (YES | NO): ")
    if creare_input.__eq__('YES'):
        extract_data()

    show_menu()



