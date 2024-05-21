import os
import numpy as np
import pandas as pd
import torch
import pickle
import matplotlib.pyplot as plt

from stpy.continuous_processes.gauss_procc import GaussianProcess
from mutedpy.utils.protein_operator import ProteinOperator
from mutedpy.utils.loaders.loader_basel import BaselLoader
from mutedpy.utils.sequences.sequence_utils import drop_neural_mutations, order_mutations, create_neural_mutations
from mutedpy.protein_learning.gaussian_process.regression_ards_geometry import LassoFeatureSelectorARDGeometric
from mutedpy.protein_learning.gaussian_process.regression_ards import ARDModelLearner
from mutedpy.utils.sequences.sequence_utils import generate_random_mutations,add_variant_column
from mutedpy.experiments.streptavidin.streptavidin_loader import load_first_round
from mutedpy.experiments.streptavidin.active_learning.compare_different_models import compare

"""
This script compares between splits and final model  

"""


if __name__ == "__main__":
	models = ["new_ardlasso_0_0_1_20"]

	results_folder = "../../results_strep/"
	splits = 10
	comparison_name = "plots/amino-acid-splits.png"

	comparison_mutants_file = '../AA_model/lists/random_mutants.csv'
	compare(models, comparison_name, comparison_mutants_file,results_folder, splits = splits)



