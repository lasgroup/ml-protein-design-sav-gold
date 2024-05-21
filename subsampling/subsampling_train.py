import os
from stpy.helpers.helper import cartesian
import numpy as np

final_dir = "params/"
d = {
    "data":['subsample'],
    "kernel": ['ard_matern'],
    "topk": [20],
    "feature_selector": ['lasso'],
    "aminoacid": [1],
    "rosetta": [0],
    "geometric": [0],
    "model": ["ARD"],
    "target": ["subsampling_analysis"],
    "seed": np.arange(0, 100, 1).tolist(),
    "subsample": [1., 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
    "final_dir": [final_dir],
    "restarts": [3],
    "maxiters": [200],
    "special_identifier": ["subsampling"]
}
string_type = [1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1]

script_name = "python ../../benchmark_run/run_benchmark.py "
dtype = np.asarray(["streptavidin-gold","streptavidin-gold-def","streptavidin-gold-opt"]).dtype
v = cartesian([a for a in d.values()], dtype = dtype)


def preamble(index):
    return """#!/bin/bash 
#SBATCH -n 15
#SBATCH --time=03:59:00
#SBATCH --mem-per-cpu=1000
#SBATCH --tmp=1000                        # per node!!
#SBATCH --job-name=benchmark"""+str(index)+"""
#SBATCH --output=/cluster/project/krause/mmutny/"""+str(index)+"""-subsampling.stdout
#SBATCH --error=/cluster/project/krause/mmutny/"""+str(index)+"""-subsampling.stderr"""

for counter, val in enumerate(v):
    command = script_name
    names = list(d.keys())
    for index, arg in enumerate(val):
        if string_type[index]:
            command += "--"+names[index] + "='"+arg+"' "
        else:
            command += "--"+names[index] + "=" + arg + " "
    print (command)
    f = open("jobs/job"+str(counter)+".sh","w")
    f.writelines(preamble(counter)+"\n")
    f.writelines(command)
    f.close()


