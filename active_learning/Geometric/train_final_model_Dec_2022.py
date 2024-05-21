import os
from stpy.helpers.helper import  cartesian

"""
This script was created in order to accomodate new structures generation with fixed seed and optimization to maintain reproducibility
"""

final_dir = "params/old_dec_2022_final"
d = {
    "kernel": ['ard_matern'],
    "topk": [50],
    "feature_selector" : ['lasso'],
    "aminoacid": [1],
    "rosetta": [0],
    "geometric": [1],
    "model": ["ARD"],
	"target": ["final_model"],
	"final_dir": [final_dir],
	"restarts" : [3],
	"maxiters" : [250],
	"special_identifier": ["DEC_2022_final"],
    "njobs": [8],
    "cores_split": [1],
    "splits": [20],
    "model_selection_data": ["all"],
    "project_geo": ["streptavidin-gold-old"],
}
string_type = [1,0,1,0,0,0,1,1,1,0,0,1,0,0,0,1,1]

script_name = "pythonconda ../../benchmark_run/run_benchmark.py "
v = [d[key][0] for key in d.keys()]
command = script_name
names = list(d.keys())
for index, arg in enumerate(v):
	if string_type[index]:
		command += "--"+names[index] + "='"+str(arg)+"' "
	else:
		command += "--"+names[index] + "=" + str(arg) + " "
print (command)
os.system(command)
