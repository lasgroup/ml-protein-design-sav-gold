import pandas as pd

from mutedpy.experiments.streptavidin.active_learning.experiment_design import generate_elems, generate_all_combination
from mutedpy.experiments.streptavidin.active_learning.calculate_predictions import generate_all_pairs
from mutedpy.experiments.streptavidin.active_learning.compare_different_models import load_model
from mutedpy.experiments.streptavidin.streptavidin_loader import load_total
from mutedpy.protein_learning.safety.safety_model import ResidueBlockSafetyModel, GPSafetyModel


if __name__ == "__main__":
    positions = [111, 112, 118, 119, 121]
    parent = 'TSNAK'
    xtest, new_mutants = generate_all_pairs(positions,parent)

    unsafe = {0: ['W'], 1: ['C', 'W'], 2: ['W'], 3: ['W'], 4: ['W']}
    safety_model = ResidueBlockSafetyModel(unsafe)

    new_mutants['safety_1'] = safety_model.query_safe(xtest).int()

    GP, embed = load_model("OD_model/params/final_model_params.p", "OD safety")
    safety_model_od = GPSafetyModel(GP, embed)

    new_mutants['safety_2'] = safety_model_od.query_safe(xtest).int()

    new_mutants.to_csv("OD_model/lists/safety.csv")