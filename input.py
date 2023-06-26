import numpy as np

def getInput(path):
    return np.loadtxt(path, delimiter=';')
