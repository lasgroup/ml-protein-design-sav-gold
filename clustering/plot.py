import matplotlib.pyplot as plt
import numpy as np
import pickle
from matplotlib.colors import ListedColormap
from matplotlib.cm import ScalarMappable
import pandas as pd
import seaborn as sns
from mutedpy.utils.sequences.sequence_utils import generate_all_combination, generate_random_mutations,from_mutation_to_variant,from_variant_to_integer
from mutedpy.experiments.streptavidin.streptavidin_loader import load_total, load_full, load_last_round,load_suggestion
from mutedpy.utils.sequences.sequence_utils import from_mutation_to_variant
variants = from_mutation_to_variant(generate_all_combination([111, 112, 118, 119, 121], 'TSNAK'))
from mutedpy.experiments.streptavidin.streptavidin_loader import tobias_colors


x, y, d = load_full()
x4, y4, d4 = load_last_round()
d4 = d4[d4['round'] == "4th"]
d4['variant'] =d4['mutant']

print (d4)
sug_dt = load_suggestion()

measured = pd.concat((d,d4))
seqs = pd.DataFrame({'variant':variants})
print (measured['variant'].head(10))
print (seqs['variant'].head(10))
measured = measured.groupby('variant').agg({'LogFitness': 'mean', 'class': 'first','round':'first'}).reset_index()
intersection_df = pd.merge(seqs, measured[['variant','LogFitness','class','round']], on='variant', how='outer')
intersection_df['class'] = intersection_df['class'].values.astype(str)
print (intersection_df)



# separate subsample
not_eval_mask = intersection_df['class']=='nan'
intersection_df.loc[not_eval_mask,'class'] = 'not_eval'
df_sample = intersection_df[not_eval_mask].sample(frac=0.05)
mask = intersection_df.index.isin(df_sample.index)
intersection_df.loc[mask,'class']= 'nan'

perplexity = 150
X_embedded = pickle.load(open('embAA-cluster_'+str(perplexity)+'.pickle', 'rb'))




intersection_df['LogFitness'] = intersection_df['LogFitness']

intersection_df['size'] = 1.
intersection_df['round'] = intersection_df['class'].apply(lambda x: x[0:3])
intersection_df.loc[mask,'name'] = "Not Evaluated"

print (intersection_df['class'].unique())

tobetriangles = ['2nd_safe','2nd_balanced','2nd_informative','2nd_optimistic-diverse'] # second round
tobesquares = ['1st-2site','1st-5site'] # first round
tobestars = []
for s in intersection_df['class'].unique():
    if "3rd" in s or "4th" in s:
        tobestars.append(s)
#tobestars = ['3rd_safe','3rd_balanced','3rd_informative','3rd_optimistic-diverse'] # third round

mask_triangle = intersection_df['class'].isin(tobetriangles)
mask_dots = intersection_df['class'].isin(tobesquares)
mask_starts = intersection_df['class'].isin(tobestars)

intersection_df.loc[mask_triangle,'size'] = 3.
intersection_df.loc[mask_triangle,'name'] = 'Exploration'

intersection_df.loc[mask_dots,'size'] = 5.
intersection_df.loc[mask_dots,'name'] = 'Initial Screen'


intersection_df.loc[mask_starts,'size'] = 5.
intersection_df.loc[mask_starts,'name'] = 'Exploitation'

intersection_df['x'] = X_embedded[:,0]
intersection_df['y'] = X_embedded[:,1]

intersection_df.to_csv("coordinates_embedding"+str(perplexity)+".csv")

to_plot = intersection_df[intersection_df['class']!='not_eval']

to_plot['size'].values.astype(int)
marker_dict = {'Exploration': '^', 'Initial Screen': 's', "Not Evaluated": '.','Exploitation': '*'}
hue_dict = {'Exploration': None, 'Initial Screen': None, 'Not Evaluated': None,'Exploitation': None}
color_dict = {'Exploration': tobias_colors['GREEN'], 'Initial Screen': tobias_colors['RED'], 'Not Evaluated': tobias_colors['GRAY'],'Exploitation': tobias_colors['BLUE']}
palette_dict= {'Exploration': tobias_colors['GREEN'], 'Initial Screen': tobias_colors['RED'], 'Not Evaluated': tobias_colors['GRAY'],'Exploitation': tobias_colors['BLUE']}
size_dict = {'Exploration': 75, 'Initial Screen':30, 'Not Evaluated': 5,'Exploitation': 75}
zorder_dict = {'Exploration': 5, 'Initial Screen':3, 'Not Evaluated': 1,  'Exploitation': 5}

f, ax = plt.subplots(figsize=(20, 20))
for key in marker_dict.keys():
    to_plot=intersection_df[intersection_df['name'] == key]
    g = sns.scatterplot(
        data=to_plot,
        x="x", y="y",s =size_dict[key], hue = hue_dict[key], color = color_dict[key],
        palette=palette_dict[key], ax = ax, marker = marker_dict[key], zorder = zorder_dict[key], legend = False)
ax.axis('off')

cmap = ListedColormap(sns.color_palette("viridis", as_cmap=True)(np.linspace(0, 1, 10)))
norm = plt.Normalize(intersection_df['LogFitness'].min(), intersection_df['LogFitness'].max())
sm = ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

# Add the colorbar
#cbar = plt.colorbar(sm, ax=ax, orientation='horizontal',shrink =0.2)
#cbar.ax.set_position([0.2, -0.1, 0.5, 0.1])
plt.title("$\log_{10}$ Relative cell-specific activity", fontsize=20)
plt.savefig('clustering_'+str(perplexity)+'.pdf',bbox_inches='tight', dpi=100)
#plt.show()


import numpy as np
from bokeh.layouts import column, gridplot
from bokeh.models import ColorBar, ColumnDataSource, LinearColorMapper, LogColorMapper
from bokeh.plotting import figure, show
from bokeh.transform import transform
from bokeh.plotting import figure, show, output_file
from bokeh.transform import factor_cmap, factor_mark
from bokeh.models import HoverTool
from collections import OrderedDict
import bokeh
import torch


#subsampled not sampled#
not_eval_mask = intersection_df['class']=='nan'
intersection_df.loc[not_eval_mask,'class'] = 'not_eval'

df_sample = intersection_df[not_eval_mask].sample(frac=0.05)

mask = intersection_df.index.isin(df_sample.index)
intersection_df.loc[mask,'class']= 'nan'
colors = np.array([[g, g, 150] for g in 30 + 2 * y], dtype="uint8")
LABELS = intersection_df['class'].unique().tolist()

print (LABELS)

LABELS.remove("not_eval")
marker_map = {'1st':'circle', '2nd':'triangle','4th':'hex', '3rd':'square','nan':'star'}
MARKERS = [marker_map[key[0:3]] for key in LABELS]
output_file("clustering.html", title="streptavidin-clustering")
TOOLS = "hover,pan,wheel_zoom,zoom_in,zoom_out,box_zoom,undo,redo,reset,tap,save,"
mapper = LinearColorMapper(palette="Viridis256", low=float(torch.min(y).numpy()), high=float(torch.max(y).numpy()))

labels = intersection_df['class'].values
p = figure(tools=TOOLS)
y = intersection_df['LogFitness'].values
variants = intersection_df['variant'].values
for label, marker in zip(LABELS, MARKERS):
    mask = labels == label
    d2 = dict(
        x1=X_embedded[mask, 0],
        x2=X_embedded[mask, 1],
        y=y[mask],
        labels=labels[mask],
        variant=variants[mask]
    )

    source = ColumnDataSource(d2)

    p.scatter('x1','x2',source= source, fill_alpha=0.8,size = 8,
          line_color=None,color=transform('y', mapper),
          marker=marker,legend_label=label)

color_bar = ColorBar(color_mapper=mapper, padding=0,
                         ticker=p.xaxis.ticker, formatter=p.xaxis.formatter)

hover = p.select(dict(type=HoverTool))
hover.tooltips = OrderedDict([("variant", "@variant"), ("label", "@labels"), ("logFitness", "@y")
                             ])

p.add_layout(color_bar, 'below')
p.legend.location = "top_left"
p.legend.title = "labels"
p.legend.click_policy="hide"
p.plot_height = 800
p.plot_width = 1400
show(p)


