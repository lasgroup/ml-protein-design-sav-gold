import os
import numpy as np
import pandas as pd
import torch
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from stpy.continuous_processes.gauss_procc import GaussianProcess
from mutedpy.utils.protein_operator import ProteinOperator
from mutedpy.utils.loaders.loader_basel import BaselLoader
from mutedpy.utils.sequences.sequence_utils import drop_neural_mutations, order_mutations, create_neural_mutations
from mutedpy.protein_learning.gaussian_process.regression_ards_geometry import LassoFeatureSelectorARDGeometric
from mutedpy.protein_learning.gaussian_process.regression_ards import ARDModelLearner
from mutedpy.utils.sequences.sequence_utils import generate_random_mutations,generate_all_combination,add_variant_column,hamming_distance
from mutedpy.experiments.streptavidin.active_learning.compare_different_models import load_model_and_mean_std, load_model
from mutedpy.experiments.streptavidin.streptavidin_loader import load_second_round,load_first_round,load_total

from stpy.helpers.helper import estimate_std

def update_model_and_save(old_dict_location, data, new_dict_location):
    """
    :param model_params:
    :param data: (x,y,s) blocks like this
    :return:
    """
    GP, embed, model_dict = load_model(old_dict_location,"fit, before update",model_params_return=True)
    xx =[]
    yy =[]
    Sigmas = []
    for x,y,s in data:
        n = x.size()[0]
        Sigma = s*torch.eye(n,dtype=torch.double)
        phi = embed(x)
        xx.append(phi)
        yy.append(y)
        Sigmas.append(Sigma)

    x = torch.vstack(xx)
    y = torch.vstack(yy)
    Sigma = torch.block_diag(*Sigmas)
    print(x.size(), y.size(), Sigma.size())

    print ("Fitting to the new data.")
    GP.fit_gp(x,y,Sigma = Sigma)

    model_dict['x'] = x
    model_dict['y'] = y
    model_dict['Sigma'] = Sigma
    model_dict['feature_loader'].close()
    with open(new_dict_location, "wb") as f:
        pickle.dump(model_dict, f)

    return GP

if __name__ == "__main__":

    # load old data
    x,y,dts = load_first_round()
    # load new data
    x2, y2, dts2 = load_second_round()
    sigma_std1 = estimate_std(x, y)
    sigma_std2 = estimate_std(x2, y2)

    print (sigma_std1)
    print (sigma_std2)

    #### update geometric
    data = [(x,y,sigma_std1),(x2,y2,sigma_std2)]
    # old_dict_location = "../active_learning/Geometric/params/final_model_params.p"
    # new_dict_location = "Geometric/params/final_model_params_Oct_26.p"
    # update_model_and_save(old_dict_location,data,new_dict_location)

    old_dict_location = "../active_learning/Geometric/params/dec_2022_finalfinal_model_params.p"
    new_dict_location = "Geometric/params/final_model_params_Jan_05.p"
    update_model_and_save(old_dict_location,data,new_dict_location)

    #### update animo-acid
    # data = [(x, y, sigma_std1), (x2, y2, sigma_std2)]
    # old_dict_location = "../active_learning/AA_model/params/final_model_params.p"
    # new_dict_location = "AA_model/params/final_model_params.p"
    # update_model_and_save(old_dict_location, data, new_dict_location)
