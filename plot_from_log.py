import sys
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np


def readlog(filename):
    f = open(filename, "r")
    lines = f.readlines()
    result = []
    for line in lines:
        parts = line.split("     ")
        print("**line:",line," parts: ",len(parts))
        if parts and len(parts) > 2 and parts[1].startswith("Loss: "):
            loss = parts[1].replace("Loss: ", "")
            floatloss = float(loss)
            result.append(floatloss)
    return result



def plot2d(x, y, xlabel=None, ylabel=None, filenameForSave=None):
    plt.plot(x, y)
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    
    if filenameForSave:
        plt.savefig(filenameForSave)
    plt.show()


args = sys.argv

if args and len(args) > 1:
    args = args[1:]
    input_filename = ""
    output_filename = ""
    i = 0
    num = len(args)
    while i < num:
        arg = args[i]
        if arg == "-i":
            if i+1 < num:
                input_filename = args[i+1]
            else:
                break
        elif arg == "-o":
            if i+1 < num:
                output_filename = args[i+1]
            else:
                break
        i = i + 1
    
    if input_filename and output_filename:
        loss = readlog(input_filename) #("./snapshots/davis_240x427s/log.txt")
        x = np.arange(len(loss))
        plot2d(x, loss, "iter", "loss", output_filename) # "loss_hzfurgbd_120x160s.png")