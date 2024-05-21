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
from mutedpy.experiments.streptavidin.streptavidin_loader import load_first_round
from mutedpy.experiments.streptavidin.active_learning.compare_different_models import compare

models = [
		  "new_ardlasso_0_1_1_20",
            "new_ardlasso_0_0_1_20"
		  ]


splits = 5
comparison_name = "plots/geometric-vs-amino-acid.png"

comparison_mutants_file = 'lists/random_mutants.csv'
results_folder = "../../results_strep/"
compare(models, comparison_name, comparison_mutants_file, results_folder, splits = splits)