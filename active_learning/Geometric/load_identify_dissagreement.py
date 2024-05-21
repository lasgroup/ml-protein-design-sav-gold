import numpy as np
import torch
import pandas as pd
from mutedpy.utils.sequences.sequence_utils import generate_random_mutations,generate_all_combination,add_variant_column,hamming_distance
from mutedpy.utils.loaders.loader_basel import BaselLoader


# load predictions of the basic model
prediction_loc = "../AA_model/predictions.csv"
dts2 = pd.read_csv(prediction_loc, index_col = False)
del dts2['Unnamed: 0']

dts2 = dts2.sort_values('std', ascending=True)
dts2 = dts2[['Mutation','std','variant','mean']].head(50000)


filename = "../../../../data/streptavidin/5sites.xls"
loader = BaselLoader(filename)
dts = loader.load()
filename = "../../../../data/streptavidin/2sites.xls"
loader = BaselLoader(filename)
total_dts = loader.load(parent='SK', positions=[112, 121])
total_dts = loader.add_mutations('T111T+N118N+A119A', total_dts)
total_dts = pd.concat([dts, total_dts], ignore_index=True, sort=False)

ham = lambda x: hamming_distance(x, total_dts)
dts2['Hamming'] = dts2['variant'].apply(ham)
dts2 = dts2[dts2['Hamming']>0]
dts2.to_csv("lists/AA_model_sorted_by_std.csv")