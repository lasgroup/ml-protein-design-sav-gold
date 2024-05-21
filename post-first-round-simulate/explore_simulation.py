import torch
import copy
import numpy as np
import argparse
import pickle
from mutedpy.utils.sequences.sequence_utils import generate_random_mutations, generate_all_combination, \
    add_variant_column, hamming_distance
from mutedpy.experiments.streptavidin.active_learning.compare_different_models import load_model_and_mean_std, \
    load_model
from mutedpy.utils.protein_operator import ProteinOperator
import pandas as pd
import ray

# load AA model after initial round

def generate_all_pairs(positions, parent):
    Op = ProteinOperator()
    new_mutants = generate_all_combination(positions, parent)
    new_mutants = pd.DataFrame(data=new_mutants, columns=['Mutation'])
    new_mutants = add_variant_column(new_mutants)
    xtest = torch.from_numpy(Op.translate_mutation_series(new_mutants['variant']))
    return xtest, new_mutants

# set up arguments
parser = argparse.ArgumentParser(description='Generate a database.')
parser.add_argument('--seed', action='store', help='seed', default=0, type = int)
parser.add_argument('--n', action='store', help='seed', default=720, type = int)

args = parser.parse_args()

# loop over repeats
repeats = 20
positions = [111, 112, 118, 119, 121]
parent = 'TSNAK'
n = args.n
stds = []

models = ["../active_learning/AA_model/params/dec_2022_finalfinal_model_params.p",
"../active_learning/AA_model/params/final_model_params.p"]

for m, model_params in enumerate(models):
    xtest, new_mutants = generate_all_pairs(positions, parent)
    GP, embed = load_model(model_params, "log", vintage=True)
    GP.max_size = 40000
    # sample random points
    ind = np.arange(0, len(new_mutants), 1)
    np.random.seed(args.seed)
    selected = np.random.choice(ind, n)
    x_sel = xtest[selected, :]
    print("sampled.")
    y_sel = GP.mean_std(embed(x_sel))[0]
    print("evaluated.")
    GP.add_data_point(embed(x_sel), y_sel)
    print("added.")
    # generate all possible predictions with std
    print("predicted.")
    mean, std = GP.mean_std(embed(xtest))
    new_mutants['std'] = std.view(-1)
    new_mutants['mean'] = mean.view(-1)
    new_mutants.to_csv("/cluster/project/krause/mmutny/stds"+str(args.seed)+"_model"+str(m)+"_"+str(n)+".csv")
