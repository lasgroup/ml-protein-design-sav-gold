import os
from stpy.helpers.helper import  cartesian
import numpy as np

final_dir = "params/"
d = {
    "kernel": ['ard_matern'],
    "topk": [20],
    "transformation": ['log'],
    "special_identifier": ["bench_noise_Jan_2024_"],
    "feature_selector" : ['lasso'],
    "aminoacid": [1],
    "rosetta": [0],
    "geometric": [0],
    "additive" : [0],
    "model": ["ARD"],
	"final_dir": [final_dir],
	"restarts" : [3],
	"maxiters" : [250],
	"cores": [15],
	"cores_split":[4],
	"splits": [15],
	"njobs": [5],
    "model_selection_data": ["train"],
    "prespecified_sigma": [0.15,0.015,0.25,0.05,0.1,0.2,0.3,1.5,1.0,0.5,None],

}
string_type = [1,0,1,1,1,0,0,0,0,1,1,0,0,0,0,0,0,1,0,0]
script_name = "cd ../ && python run_benchmark.py "
dtype = np.asarray(["streptavidin-gold","streptavidin-gold-def","streptavidin-gold-opt"]).dtype
v = cartesian([a for a in d.values()], dtype = dtype)

def preamble(index):
    return """#!/bin/bash 
#SBATCH -n 60
#SBATCH --time=48:59:00
#SBATCH --mem-per-cpu=1000
#SBATCH --tmp=1000                        # per node!!
#SBATCH --job-name=benchmark"""+str(index)+"""
#SBATCH --output=/cluster/project/krause/mmutny/"""+str(index)+"""-benchmark_run.stdout
#SBATCH --error=/cluster/project/krause/mmutny/"""+str(index)+"""-benchmark_run.stderr"""


for counter, val in enumerate(v):
    command = script_name
    names = list(d.keys())
    for index, arg in enumerate(val):
        if string_type[index]:
            command += "--"+names[index] + "='"+arg+"' "
        else:
            command += "--"+names[index] + "=" + arg + " "
    print (command)
    f = open("job_files_noise/job"+str(counter)+".sh","w")
    f.writelines(preamble(counter)+"\n")
    f.writelines(command)
    f.close()


