import torch
import numpy as np
import matplotlib.pyplot as plt
from mutedpy.protein_learning.active_learning.generate_predictions import load_model

params = "AA_model/params/final_model_params.p"
GP, embed, model_params = load_model(params, vintage = True, model_params_return=True)
print (model_params)