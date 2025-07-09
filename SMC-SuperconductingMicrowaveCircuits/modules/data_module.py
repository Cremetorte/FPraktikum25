# Read data from measurement file
# File has a certain number of columns and multiple measurement sets
# Sets are separated by a newline
# Single title line and ', ' field delimeter

import numpy as np
import pandas as pd
import os

pi = np.pi

def readdat_pd(filename, delim=', ', nlines=None):
    with open(filename, 'r') as f:
        # variables = {}
        # ivar = 0
        # mylists = {}
        # col = []
        # currentvarname = ""

        line = f.readline()
        line = line.strip("\n").strip("# ")
        #        print(line)
        names = line.split(delim)
        print(names)

        block = []
        arrayofframes = []
        nblocks = 0

        for point in f:
            if point == '\n':
                block = np.asarray(block)
                block = block.T
                newframe = pd.DataFrame()
                for name, dat in zip(names, block):
                    newframe[name] = dat
                arrayofframes.append(newframe)
                block = []
                nblocks += 1
                if nlines == None:
                    continue
                elif nblocks < nlines:
                    continue
                else:
                    break
            point = [float(x) for x in point.strip('\n').split(delim)]
            block.append(point)
        if len(block) != 0:
            block = np.asarray(block)
            block = block.T
            newframe = pd.DataFrame()
            for name, dat in zip(names, block):
                newframe[name] = dat
            arrayofframes.append(newframe)
        return arrayofframes

def save_params_to_csv(params, data_file):
    data = []

    for name, param in params.items():
        data.append({
            'name': name,
            'value': param.value,
            'stderr': param.stderr,
            'min': param.min,
            'max': param.max,
            'vary': param.vary
        })

    # Convert to DataFrame and save to CSV
    df = pd.DataFrame(data)

    os.makedirs('../results', exist_ok=True)
    os.makedirs('../results/kerr_analysis_fit_result/', exist_ok=True)
    df.to_csv("../results/kerr_analysis_fit_result/" + 'fit_result_' + data_file[-11:-4] + ".csv", index=False)
