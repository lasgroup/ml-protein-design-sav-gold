import numpy as np

def parallel_coordinates_bo(X, Y, names=None, scaling=None, fig_size=(20, 10), names_x = (None,None), names_bottom = None):
	"""
		Parallel plot graph

		X : 2D numpy array of parameters [points,parameters]
		Y : 1D numpy array of values
		names: list of names size of (parameters)
		scaling:
			"stat": statistical scaling
			None : no scaling
			(low,hig): tuple, scales to [-1,1]
		fig_size: fig size in inches
	"""
	from pandas.plotting import parallel_coordinates
	import pandas as pd
	import numpy as np
	import copy
	import matplotlib.pyplot as plt
	from sklearn.preprocessing import StandardScaler

	if scaling == "stat":
		scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
		scaler.fit(X)
		Z = scaler.transform(X)
	elif scaling is None:
		Z = X
	else:
		try:
			Z = X
			up, low = scaling
			d = X.shape[1]
			for i in range(d):
				Z[:, i] = (2 * X[:, i]) / (up[i] - low[i]) + (1.0 - 2 * up[i] / (up[i] - low[i]))
		except:
			pass

	D = np.append(Z, Y, axis=1)
	data = pd.DataFrame(D)
	data = data.sort_values(by=Z.shape[1])
	if names is not None:
		names = copy.copy(names)
		names.append(Z.shape[1])
		data.columns = names
	plt.figure(figsize=(fig_size))
	plt.xticks(rotation=45)
	if names_x[0] is not None:
		plt.yticks(names_x[0],names_x[1])

	ax = parallel_coordinates(data, Z.shape[1], colormap="coolwarm")
	ax.get_legend().remove()
	if names_bottom is not None:
		plt.xticks(np.arange(0,X.shape[1],1),names_bottom)

	#plt.show()

if __name__ == '__main__':
	from stpy.test_functions.protein_benchmark import ProteinBenchmark
	from stpy.test_functions.protein_benchmark import ProteinOperator
	import matplotlib.pyplot as plt
	from mutedpy.utils.loaders.loader_basel import BaselLoader
	import pandas as pd

	Operator = ProteinOperator()
	filename = "../../data/streptavidin/5sites.xls"
	loader = BaselLoader(filename)
	dts = loader.load()

	filename = "../../data/streptavidin/2sites.xls"
	loader = BaselLoader(filename)
	dts2 = loader.load(parent = 'SK', positions = [112,121])
	dts2 = loader.add_mutations('T111T+N118N+A119A',dts2)

	dts = pd.concat([dts, dts2], ignore_index=True, sort=False)

	print (dts['variant'])

	f = lambda x: np.array(list(map(int,[Operator.dictionary[a] for a in list(str(x))]))).reshape(-1,1)
	xtest = np.concatenate(dts['variant'].apply(f).values, axis = 1).T
	print (xtest.shape)
	print (xtest)
	ytest = dts['Fitness'].values.reshape(-1,1)
	print (ytest.shape)
	print (ytest)
	real_names = [Operator.real_names[Operator.inv_dictionary[a]] for a in np.arange(0,20,1)]
	print (real_names)
	parallel_coordinates_bo(xtest,ytest, names_x = (np.arange(0,20,1),real_names), names_bottom=[112,112,118,119,121])

	plt.savefig("streptavidin.png",dpi = 300)
	plt.show()

