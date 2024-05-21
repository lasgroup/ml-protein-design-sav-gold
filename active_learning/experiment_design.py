import os
import numpy as np
import pandas as pd
import torch
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from stpy.continuous_processes.gauss_procc import GaussianProcess
from mutedpy.utils.protein_operator import ProteinOperator
from mutedpy.protein_learning.regression.regression_ards import ARDModelLearner
from mutedpy.utils.sequences.sequence_utils import generate_random_mutations,generate_all_combination,add_variant_column,hamming_distance
from mutedpy.experiments.streptavidin.streptavidin_loader import load_second_round, load_first_round, load_total
from mutedpy.experiments.streptavidin.active_learning.compare_different_models import load_model

from dppy.finite_dpps import FiniteDPP


def filter_by_safety(selected, safety_columns_names):
    final_mask = None
    if safety_columns_names is not None:
        for saf in safety_columns_names:
            mask = selected[saf] == 1
            if final_mask is None:
                final_mask = mask.values
            else:
                final_mask = np.logical_and(final_mask, mask.values)
        selected = selected[final_mask]
    return selected

def generate_elems(dts, N, type = 'lcb', name = 'safe', oversample_for_diversity = 5, already_known_variants = None,
                   safety_column_names = None, diverse = True, GP = None, embed = None, jitter = 0.):

    # remove already known
    print("Starting with:",len(dts))
    mask = ~dts['variant'].isin(already_known_variants['variant'])
    selected = dts[mask]
    print(len(selected))
    rng = np.random.RandomState(1)
    Op = ProteinOperator()
    selected = filter_by_safety(selected, safety_column_names)

    if diverse and GP is not None:
        optimistic_mutants = selected.sort_values(by=[type], ascending=False).head(N*oversample_for_diversity)

        x_optimistic = embed(torch.from_numpy(Op.translate_mutation_series(optimistic_mutants['variant'])))
        optimistic_mean, K_opt = GP.mean_std(x_optimistic, full=True)
        K_opt = K_opt + torch.eye(K_opt.size()[0], dtype = torch.double)*jitter
        DPP = FiniteDPP('likelihood', **{'L': K_opt.detach().numpy()})

        sample_optimistic = DPP.sample_exact_k_dpp(size=N, random_state=rng, model = "KuTa12")
        out = optimistic_mutants.iloc[sample_optimistic, :]

    else:
        optimistic_mutants = selected.sort_values(by=[type], ascending=False).head(N)
        out = optimistic_mutants

    out.loc[:,'category'] = name
    return out

def generate_informative(dts, N, name = 'safe', steps_by = 10000, already_known_variants = None, safety_column_names = None, GP = None, embed = None):
    Op = ProteinOperator()
    # remove already known
    mask = dts['variant'].isin(already_known_variants['variant'])
    selected = dts[mask]
    rng = np.random.RandomState(1)

    selected = filter_by_safety(selected, safety_column_names)

    xtest = torch.from_numpy(Op.translate_mutation_series(selected['variant']))
    phitest = embed(xtest)
    most_informative_indices = []
    for j in range(N):
        print("Selecting", j, "/", N)
        sample = np.random.choice(np.arange(0, phitest.size()[0]), steps_by)
        phi_sel = phitest[sample, :]
        _, std = GP.mean_std(phi_sel, reuse=True)

        most_informative_std = float(torch.max(std.view(-1)))
        most_informative_index = sample[int(torch.argmax(std.view(-1)))]

        print("selected:", most_informative_index, most_informative_std)

        phinew = phitest[most_informative_index, :].view(1, -1)

        GP.add_data_point(phinew, torch.Tensor([[0.]]).double())
        most_informative_indices.append(int(most_informative_index))

    info = selected.loc[most_informative_indices, :]
    info.loc[:,'category'] = name
    return info


if __name__ == "__main__":
    # params
    N = 120

    # load total data
    x, y, dts = load_total()

    ### generate AA safe and mean
    model_params = "../active_learning_2/AA_model/params/final_model_params.p"
    GP, embed = load_model(model_params, "second round fit")

    # load predictions
    predictions = pd.read_csv("../active_learning_2/AA_model/lists/predictions-aa.csv")

    safe = generate_elems(predictions,N,already_known_variants=dts, type = 'lcb', name = 'safe', GP= GP, embed=embed)
    dts = pd.concat([dts,safe])
    balanced = generate_elems(predictions, N,already_known_variants=dts, type='mean', name='balanced', GP=GP, embed=embed)

    aa_model_suggestions = pd.concat([safe,balanced])
    aa_model_suggestions.to_csv("../active_learning_2/AA_model/lists/safe+balanced-aa.csv")
    aa_model_suggestions.to_html("../active_learning_2/AA_model/lists/safe+balanced-aa.html")

    # generate Geometric safe and mean