import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from os.path import join as pjoin
from os.path import isfile
from os import listdir
from utils import *
import sklearn as sk 
from scipy.spatial.distance import jensenshannon
from scipy import stats



muon_folder = "C:\\Users\\lovat\\Desktop\\MyFiles\\Unif\Master's Thesis\\Data\\muon"
data_folder = "C:\\Users\\lovat\\Desktop\\MyFiles\\Unif\Master's Thesis\\Data\\data"
neutrino_folder = "C:\\Users\\lovat\\Desktop\\MyFiles\\Unif\Master's Thesis\\Data\\neutrino"
distribution_plot_folder = "C:\\Users\\lovat\\Desktop\\MyFiles\\Unif\Master's Thesis\\Plots\\distributions"
proportion_plot_folder = "C:\\Users\\lovat\\Desktop\\MyFiles\\Unif\Master's Thesis\\Plots\\proportions"
general_plot_folder = "C:\\Users\\lovat\\Desktop\\MyFiles\\Unif\Master's Thesis\\Plots\\general"

dataframes = []
for filename in listdir(muon_folder):
    file = pjoin(muon_folder, filename)
    print(file)
    dataframes.append(pd.read_hdf(file))
X_muon_raw = pd.concat(dataframes)

dataframes = []
for filename in listdir(neutrino_folder):
    if filename[:4] == "numu":
        file = pjoin(neutrino_folder, filename)
        print(file)
        dataframes.append(pd.read_hdf(file))
X_numu_raw = pd.concat(dataframes)

initial_columns = X_muon_raw.columns
useful_columns = []
for column in initial_columns:
    if "shower" not in column and "mc" not in column:
        useful_columns.append(column)

X_muon = X_muon_raw[useful_columns]
X_numu = X_numu_raw[useful_columns]
X_muon["type"] = 0.0
X_numu["type"] = 1.0

dataframes = [X_muon, X_numu]
X = pd.concat(dataframes)

X = X[:][X["n_trig_hits"] > 20]
X = X[:][X["n_trig_lines"] >= 2]
X = X[:][X["bestmuon_dz"] < -0.2]

y = X["type"]
x = X.drop(columns=["weight_sig", "weight_bkg", "run", "type", "bestmuon_dz", "bestmuon_dx", "bestmuon_dy"])
ind = (np.abs(stats.zscore(x)) < 3).all(axis=1)
x = x[ind]
y = y[ind]
X = X[ind]
x = (x-x.mean())/x.std()
w = np.where(X["type"] == 0, X["weight_bkg"], X["weight_sig"])

optimal_weights = getOptimalAxis(x, y, w)
X["optimal_feature"] = computeOptimalFeature(x, x.columns, optimal_weights)
#displayHistogram(X, "optimal_feature", nbins=100, useWeights=True)
features = x.columns.tolist()
features.append("optimal_feature")
features = np.array(features)
dic, JSscores = getJensenShannonScores(X, features, w, distribution_plot_folder)
MIscores = getMIScores(X, features, "type", 5)
CORRscores = np.abs(getCorrScores(X, features, "type", w))
features = features[np.argsort(JSscores)]
MIscores = MIscores[np.argsort(JSscores)]
CORRscores = CORRscores[np.argsort(JSscores)]
JSscores = np.sort(JSscores)
features = np.flip(features)
JSscores = np.flip(JSscores)
MIscores = np.flip(MIscores)
CORRscores = np.flip(CORRscores)

x = np.arange(len(features))  # the label locations
width = 0.25  # the width of the bars
multiplier = 0

fig, ax = plt.subplots(layout='constrained')

allScores = [("JS", JSscores), ("MI", MIscores), ("CORR", CORRscores)]

for attribute, measurement in allScores:
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement/np.max(measurement), width, label=attribute)
    ax.bar_label(rects, padding=3)
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Normalized score')
ax.set_title('Feature to target scores')
ax.set_xticks(x + width, features)
ax.legend(loc='upper left', ncols=3)
ax.set_ylim(0, 1.1)

plt.show()

for i in range(len(features)):
    print(features[i], "\t", round(JSscores[i], 5), "\t", round(MIscores[i], 5))


getCorrelationMatrix(X, features, w, general_plot_folder)

"""
computeProbabilityBasedFeature(X, dic, features=features[:10])
displayHistogram(X, "muon_score", nbins=100, useWeights=True)
displayHistogram(X, "neutrino_score", nbins=100, useWeights=True)
displayHistogram(X, "score", nbins=100, useWeights=True)
"""
