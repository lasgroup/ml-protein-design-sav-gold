from mutedpy.protein_learning.data_splits import create_splits
from mutedpy.experiments.streptavidin.streptavidin_loader import load_total,load_od_data

if __name__ ==  "__main__":
    x,y,dts = load_od_data()
    create_splits(x, y, 20, 100, "splits/random_splits.pt")
