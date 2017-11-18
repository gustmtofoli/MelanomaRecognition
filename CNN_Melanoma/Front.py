import InfoMessages as Info
import tkinter as tk
from tkinter import filedialog


def _validationPath(path):
    if path is None:
        return Info.errorMessage()


def _validationType(type):
    if type == "" or type == None:
        return Info.errorMessage()


def chooseFolder(type):
    _validationType(type)
    tk.Tk().withdraw()
    path = filedialog.askdirectory()
    _validationPath(path)
    Info.infoFolder(path, type)
    return path