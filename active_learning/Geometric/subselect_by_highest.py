#%%

import os
import numpy as np
import pandas as pd
import torch
import pickle
import matplotlib.pyplot as plt

from stpy.continuous_processes.gauss_procc import GaussianProcess
from stpy.test_functions.protein_benchmark import ProteinOperator
from mutedpy.utils.loaders.loader_basel import BaselLoader
from mutedpy.utils.sequences.sequence_utils import drop_neural_mutations, order_mutations, create_neural_mutations
from mutedpy.protein_learning.gaussian_process.regression_ards_geometry import LassoFeatureSelectorARDGeometric
from mutedpy.protein_learning.gaussian_process.regression_ards import ARDModelLearner
from mutedpy.utils.sequences.sequence_utils import generate_random_mutations,add_variant_column
from stpy.continuous_processes.nystrom_fea import NystromFeatures
from stpy.continuous_processes.kernelized_features import KernelizedFeatures
from stpy.embeddings.polynomial_embedding import CustomEmbedding

models = ["squaredgeometric_AA_ard_lasso_features_50_kernel_ard_matern"]
pars = [{'data_folder':'../../data/','features_G': True,'features_V': False,
		 'kernel':'ard_matern','topk':50, 'restarts':5,"maxiter":500}]#,{'data_folder':'../data/'}]
results_folder = "../../results_strep/"
splits = 10


Op = ProteinOperator()


filename = "../../../../data/streptavidin/5sites.xls"
loader = BaselLoader(filename)
dts = loader.load()

filename = "../../../../data/streptavidin/2sites.xls"
loader = BaselLoader(filename)
total_dts = loader.load(parent = 'SK', positions = [112,121])
total_dts = loader.add_mutations('T111T+N118N+A119A',total_dts)

total_dts = pd.concat([dts, total_dts], ignore_index=True, sort=False)
total_dts = create_neural_mutations(total_dts)
total_dts['LogFitness'] = np.log10(total_dts['Fitness'])

x = torch.from_numpy(Op.translate_mutation_series(total_dts['variant']))
y = torch.from_numpy(total_dts['LogFitness'].values).view(-1, 1)

# load the questioned mutants
new_mutants = pd.read_csv("AA_model_sorted_by_std.csv")
new_mutants = add_variant_column(new_mutants)
xtest = torch.from_numpy(Op.translate_mutation_series(new_mutants['variant']))

xtotal = torch.vstack((x,xtest))

dummy_model = LassoFeatureSelectorARDGeometric(**pars[0])
dummy_model.preload()
res = []
embed = lambda x: torch.hstack([e.embed(x) for e in dummy_model.list_of_embeddings])[:, feature_mask]
d = pickle.load(open('final_model_params.p', "rb"))
k = d['kernel_object']
phi =d['x']
yy = d['y']
s = d['noise_std']
feature_mask = d['feature_mask']

GP = GaussianProcess(kernel=k, s=s)


GP.fit_gp(phi, yy)
phitest = embed(xtest)
mu,std = GP.mean_std(phitest)

# preidctions
new_mutants['mean_2'] = mu
new_mutants['std_2'] = std
new_mutants['ucb_2'] = mu + 2*std
new_mutants = new_mutants.sort_values(by=['mean_2'], ascending = False).head(10000)

m = 1024
d = 5

nys = NystromFeatures(k, m = 1024,approx = "svd")
nys.fit_gp(xtotal,torch.zeros(size=(xtotal.size()[0],1)))
emb = CustomEmbedding(d,nys.embed,m)

GP2 = KernelizedFeatures(nys,m)
GP2.fit_gp(x,y)

# all data
new_mutants.to_csv("top-values-geometric.csv")


#%%



#%%



