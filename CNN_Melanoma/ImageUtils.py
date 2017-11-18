import InfoMessages as Info
from Preprocessing import load_images
import tkinter as tk
from tkinter import filedialog
import numpy as np


def _validationList(list):
    if len(list) == 0:
        return Info.errorMessage()


# def _validationPath(path):
#     if path is None:
#         return Info.errorMessage()


def _validationType(type):
    if type == "" or type == None:
        return Info.errorMessage()


# def chooseFolder(type):
#     _validationType(type)
#     tk.Tk().withdraw()
#     path = filedialog.askdirectory()
#     _validationPath(path)
#     Info.infoFolder(path, type)
#     return path


def loadImages(path, type):
    _validationType(type)
    image_List = load_images(path)
    # image_List.shape
    _validationList(image_List)
    image_List = np.array(image_List)
    image_List.shape
    # image_List = np.reshape(image_List,[-1, int(np.prod(image_List.shape[1:]))])
    # image_List.shape
    Info.infoImages(image_List)
    return image_List