import torch
import numpy as np

import torch
import copy
import numpy as np
import argparse
import pickle
from mutedpy.utils.sequences.sequence_utils import generate_random_mutations, generate_all_combination, \
    add_variant_column, hamming_distance
from mutedpy.experiments.streptavidin.active_learning.compare_different_models import load_model_and_mean_std, \
    load_model
from mutedpy.utils.sequences.sequence_utils import from_variant_to_integer
from mutedpy.utils.protein_operator import ProteinOperator
import pandas as pd
import ray

import torch
import numpy as np
import matplotlib.pyplot as plt
from mutedpy.utils.sequences.sequence_utils import add_variant_column
from mutedpy.experiments.streptavidin.streptavidin_loader import *




# load the data
# load the third + fourth round data
x3, y3, d3 = load_third_round()
x4, y4, d4 = load_last_round()
selection = d4[d4['round'] == "4th"]
selection = pd.concat([d3, selection])
selection = selection[['variant','LogFitness','class']]
selection = selection.groupby('variant').agg({'LogFitness': 'mean', 'class': 'first'}).reset_index()

x_sel = from_variant_to_integer(selection['variant'].values)
y = torch.from_numpy(selection['LogFitness'].values).view(-1,1)

# iterate over models
# predict the hit rate - from the best prediction
fractions = np.arange(0.1,1.1,0.1)
repeats = 20
mask_actual = y > np.log10(8.625)
mean_error = []
mean_hit_rate = []


print ([fractions[i] for i in range(10) for r in range(repeats)])


for fraction in fractions:
    #hit_rate = 0
    #error = 0
    for r in range(repeats):
        try:
            filename_params = "jobs/params/subsampling"+ str(np.round(fraction,1)) + "_"+ str(r) + ".p"
            GP, embed = load_model(filename_params, "log", vintage=True)

            # predict the values
            mu, std = GP.mean_std(embed(x_sel))

            mask_predicted = mu > np.log10(8.625)

            error = torch.mean((mu - y)**2)
            #print (torch.sum(mask_actual*mask_predicted),torch.sum(mask_actual))
            hit_rate = torch.sum(mask_actual*mask_predicted)/torch.sum(mask_actual)

            print ("Frac: %f, Error:  %f, Hit rate: %f" % (fraction, float(error), float(hit_rate)))
            #hit_rate = hit_rate/repeats
            #error   = error/repeats
            mean_hit_rate.append(float(hit_rate))
            mean_error.append(float(error))
        except:
            pass


filename_params = "../active_learning/AA_model/params/final_model_params.p"
GP, embed = load_model(filename_params, "log", vintage=True)

# predict the values
mu, std = GP.mean_std(embed(x_sel))

mask_predicted = mu > np.log10(8.625)

error = torch.mean((mu - y)**2)
#print (torch.sum(mask_actual*mask_predicted),torch.sum(mask_actual))
hit_rate = torch.sum(mask_actual*mask_predicted)/torch.sum(mask_actual)

print ("Frac: %s, Error:  %f, Hit rate: %f" % ("First", float(error), float(hit_rate)))
#hit_rate = hit_rate/repeats
#error   = error/repeats
mean_hit_rate.append(float(hit_rate))
mean_error.append(float(error))


############# proportion in the batch #################

error = 0
#print (torch.sum(mask_actual*mask_predicted),torch.sum(mask_actual))
hit_rate = torch.sum(mask_actual)/float(y.size()[0])
mean_hit_rate.append(float(hit_rate))
mean_error.append(float(error))

print ("Frac: %s, Error:  %f, Hit rate: %f" % ("First", float(error), float(hit_rate)))

for index, fraction in enumerate(fractions):
    print ("Fraction: %f, Error: %f, Hit rate: %f" % (fraction, mean_error[index], mean_hit_rate[index]))

pd.DataFrame({'fraction': [fractions[i] for i in range(10) for r in range(repeats)] + ['Initial',"random"], 'error': mean_error, 'hit_rate': mean_hit_rate}).to_csv("subsampling.csv")
