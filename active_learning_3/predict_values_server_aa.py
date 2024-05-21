import torch
import numpy
from stpy.test_functions.protein_benchmark import ProteinOperator
from mutedpy.utils.sequences.sequence_utils import generate_random_mutations,add_variant_column
from mutedpy.protein_learning.embeddings.amino_acid_embedding import AminoAcidEmbedding
from mutedpy.protein_learning.featurizers.feature_loader import ProteinFeatureLoader,AddedProteinFeatureLoader
from mutedpy.experiments.streptavidin.streptavidin_loader import load_second_round, load_full, load_total
import os
from mutedpy.protein_learning.active_learning.generate_predictions import load_model
import argparse
import pandas as pd

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--input', action='store', help='input [txt]', default="prediction_jobs/input0.txt")
parser.add_argument('--output', action='store', help='input [txt]', default="output/output0.csv ")

args = parser.parse_args()

seqs = []
for line in open(args.input, "r"):
    seqs.append(line.strip("\n"))
path_dir = "../"

Op = ProteinOperator()
new_mutants = pd.DataFrame(data=seqs, columns=['Mutation'])
new_mutants = add_variant_column(new_mutants)
xtest = torch.from_numpy(Op.translate_mutation_series(new_mutants['variant']))

GP, embed ,full_dictionary = load_model(path_dir+"AA_model/params/final_model_params_Jan_05.p", model_params_return = True)
phi = embed(xtest)
mu, std = GP.mean_std(phi)
#
data = {'mean':mu.detach().numpy().reshape(-1), 'Mutation':seqs, 'std':std.detach().numpy().reshape(-1)}
dts = pd.DataFrame(data)
dts['lcb'] = dts['mean'] - 2*dts['std']
dts['ucb'] = dts['mean'] +  2*dts['std']
dts.to_csv(args.output)
