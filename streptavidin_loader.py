import torch
from mutedpy.utils.loaders.loader_basel import BaselLoader
import pandas as pd
from mutedpy.utils.sequences.sequence_utils import drop_neural_mutations, order_mutations, create_neural_mutations
import numpy as np
from mutedpy.utils.protein_operator import ProteinOperator
import os

tobias_colors = {
'BLUE' :'#619CFF',
'RED' : '#F8766D',
'GREEN' : '#00BA38',
'GRAY' : '#BEBEBE'
}
def load_structure():
    filename = os.path.dirname(__file__) + "/../../../data/streptavidin/6j6j-dimer.pdb"
    return filename

def load_first_round(transformation = 'log'):
    Op = ProteinOperator()

    filename = os.path.dirname(__file__) + "/../../../data/streptavidin/5sites.xls"
    loader = BaselLoader(filename)
    dts = loader.load()
    dts['class'] = "1st-5site"

    filename = os.path.dirname(__file__) + "/../../../data/streptavidin/2sites.xls"
    loader = BaselLoader(filename)
    total_dts = loader.load(parent='SK', positions=[112, 121])
    total_dts = loader.add_mutations('T111T+N118N+A119A', total_dts)
    total_dts['class'] = "1st-2site"

    total_dts = pd.concat([dts, total_dts], ignore_index=True, sort=False)
    total_dts = create_neural_mutations(total_dts)

    if transformation == "log":
        total_dts['LogFitness'] = np.log10(total_dts['Fitness'])
    elif transformation == "minusone":
        total_dts['LogFitness'] = total_dts['Fitness']-1
    elif transformation == "none":
        total_dts['LogFitness'] = total_dts['Fitness']
    elif transformation == "loge":
        total_dts['LogFitness'] = np.log(total_dts['Fitness'])

    x = torch.from_numpy(Op.translate_mutation_series(total_dts['variant']))
    y = torch.from_numpy(total_dts['LogFitness'].values).view(-1, 1)
    return x,y, total_dts

def load_second_round():
    Op = ProteinOperator()
    filename = os.path.dirname(__file__) + "/../../../data/streptavidin/ML1_chimeras.csv"
    new_data = pd.read_csv(filename)
    x = torch.from_numpy(Op.translate_mutation_series(new_data["variant"]))
    new_data['Fitness'] = new_data['norm_TSNAK']
    new_data['LogFitness'] = np.log10(new_data['norm_TSNAK'])
    new_data['Mutation'] = new_data['variant'].apply(lambda x: Op.mutation("TSNAK",[111,112,118,119,121],x))
    new_data = create_neural_mutations(new_data)
    del new_data['mean']
    del new_data['Hamming']
    del new_data['std']
    del new_data['lcb']
    del new_data['ucb']
    del new_data['norm_TFNAQ']
    new_data['class'] = "2nd_"+ new_data['class']

    y = torch.from_numpy(np.log10(new_data['norm_TSNAK'].values)).view(-1, 1)
    return x, y, new_data


def load_third_round():
    Op = ProteinOperator()
    filename = os.path.dirname(__file__) + "/../../../data/streptavidin/exploitation-round/ArM screening ML2 all data.csv"
    new_data = pd.read_csv(filename)
    mask = new_data['clone'] == "mutant"
    new_data = new_data[mask]
    new_data = new_data.dropna(subset=['variant'])
    x = torch.from_numpy(Op.translate_mutation_series(new_data["variant"]))
    new_data['Fitness'] = new_data['norm_TSNAK']
    new_data['LogFitness'] = np.log10(new_data['norm_TSNAK'])
    new_data['Mutation'] = new_data['variant'].apply(lambda x: Op.mutation("TSNAK", [111, 112, 118, 119, 121], x))
    new_data = create_neural_mutations(new_data)
    y = torch.from_numpy(np.log10(new_data['norm_TSNAK'].values)).view(-1, 1)

    new_data['category'] = new_data['category'].fillna('chimera')
    new_data['model'] = new_data['model'].fillna('NAN')

    new_data['class'] = "3rd_" + new_data['model'] + "_" + new_data['category']
    return x,y, new_data



def load_full():
    x1,y1,dts1 = load_first_round()
    x2,y2,dts2 = load_second_round()
    x3, y3, dts3 = load_third_round()
    x = torch.vstack([x1,x2,x3])
    y = torch.vstack([y1,y2,y3])
    return x, y, pd.concat([dts1, dts2, dts3],ignore_index=True)

def load_everything_we_have():
    return load_full()

def load_total():
    x1,y1,dts1 = load_first_round()
    x2,y2,dts2 = load_second_round()
    x = torch.vstack([x1,x2])
    y = torch.vstack([y1,y2])
    return x, y, pd.concat([dts1, dts2],ignore_index=True)

def load_last_round():
    Op = ProteinOperator()
    filename = os.path.dirname(__file__) + "/../../../data/streptavidin/last_round/Quintuple mutants results.csv"
    new_data = pd.read_csv(filename)

    new_data['Fitness'] = new_data['norm_TSNAK']
    new_data['LogFitness'] = np.log10(new_data['norm_TSNAK'])

    mask = new_data['mutant'] != "empty_vector"
    new_data = new_data[mask]
    mask = new_data['mutant'] != "tbd"
    new_data = new_data[mask]
    new_data['round'] = "4th"
    new_data['variant'] = new_data['mutant']
    new_data['Mutation'] = new_data['variant'].apply(lambda x: Op.mutation("TSNAK", [111, 112, 118, 119, 121], x))
    new_data['class'] = "4th_" + new_data['category']
    x = torch.from_numpy(Op.translate_mutation_series(new_data["variant"]))
    y = torch.from_numpy(np.log10(new_data['norm_TSNAK'].values)).view(-1, 1)

    filename = os.path.dirname(__file__) + "/../../../data/streptavidin/last_round/Extrapolation results.csv"

    extrapolation = pd.read_csv(filename)
    extrapolation['Mutation'] = extrapolation['mutant'].apply(lambda x: "+".join(x.split(" ")))

    d = pd.concat([new_data, extrapolation])

    return x,y,d

def load_od_data():
    Op = ProteinOperator()
    dts = pd.read_excel(os.path.dirname(__file__) + "/../../../data/streptavidin/raw-data/ArM screening 5 positions complete data set.xlsx")
    mask = dts['variant'].notna()
    dts = dts[mask]
    y = (torch.from_numpy(dts['OD'].values).view(-1,1) + 2)/2
    x = torch.from_numpy(Op.translate_mutation_series(dts["variant"]))
    return x,y,dts

def load_suggestion( rounds = 'all'):
    dts1 = pd.read_csv(os.path.dirname(__file__) + "/../../../data/streptavidin/active_learning/round_1/balanced.csv")
    dts1['category'] = 'balanced-aa'
    dts2 = pd.read_csv(os.path.dirname(__file__) + "/../../../data/streptavidin/active_learning/round_1/hypothesis_testing.csv")
    dts2['category'] = 'hypothesis-aa'
    dts3 = pd.read_csv(os.path.dirname(__file__) + "/../../../data/streptavidin/active_learning/round_1/informative.csv")
    dts3['category'] = 'informative-aa'
    dts4 = pd.read_csv(os.path.dirname(__file__) + "/../../../data/streptavidin/active_learning/round_1/optimistic-diverse.csv")
    dts4['category'] = 'optimistic-aa'
    dts5 = pd.read_csv(os.path.dirname(__file__) + "/../../../data/streptavidin/active_learning/round_1/safe.csv")
    dts5['category'] = 'safe-aa'

    dts = pd.concat([dts1,dts2,dts3,dts4,dts5])

    dts1_2 = pd.read_csv(os.path.dirname(__file__) + "/../../../data/streptavidin/active_learning/round_2/safe+balanced-aa.csv")
    dts1_2['model']=  'aa'
    dts2_2 = pd.read_csv(os.path.dirname(__file__) + "/../../../data/streptavidin/active_learning/round_2/safe+balanced-geo.csv")
    dts2_2['model'] = 'geo'
    dts3_2 = pd.read_csv(os.path.dirname(__file__) + "/../../../data/streptavidin/active_learning/round_2/safe+balanced-ro.csv")
    dts3_2['model'] = 'ro'

    dts_2 = pd.concat([dts1_2, dts2_2, dts3_2])
    dts_2['category'] = dts_2['category'] + "_" + dts_2['model']
    if rounds == "all":
        d = pd.concat([dts,dts_2])
        d['class'] = d['category']
        return d
    elif rounds == 1:
        dts['class'] = dts['category']
        return dts
    elif rounds == 2:
        dts_2['class'] = dts_2['category']
        return dts_2

def load_full_emb(modeltype):
    xtest = np.loadtxt(os.path.dirname(__file__) + "/../../../data/streptavidin/xtest.pt")
    phitest = np.loadtxt(os.path.dirname(__file__) + "/../../../data/streptavidin/emb"+modeltype+".pt")
    return xtest, phitest

if __name__ == "__main__":
    load_last_round()
