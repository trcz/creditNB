import numpy as np
import pandas as pd
import time

startp = time.time()

with open("credit_dataset2.csv", "rb") as myFile:
    df = pd.read_csv(myFile, sep=";", header=None)

endp = time.time()

print("Pandas read_csv timer: {0}".format(endp-startp))

startp2 = time.time()
with open("credit_dataset2.csv", "rt") as myFile:
    df2 = np.genfromtxt(myFile, delimiter=';', dtype=None)

endp2 = time.time()

print("Numpy genfromtxt timer: {0}".format(endp2-startp2))