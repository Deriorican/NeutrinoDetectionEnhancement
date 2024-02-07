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

"""dataframes = []
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


plt.hist(X["bestmuon_dz"][X["type"] == 0], 100, color = "red", alpha = 0.5, weights=X["weight_bkg"][X["type"] == 0], label = "Muons")
plt.hist(X["bestmuon_dz"][X["type"] == 1], 100,  color = "green", alpha = 0.5, weights=X["weight_sig"][X["type"] == 1], label = "Neutrinos")
plt.legend()
plt.yscale("log")
plt.ylabel("Amount of events")
plt.xlabel("Direction of track (cos[zenith])")
plt.title("Muon vs. Neutrino proportions")
plt.show()"""

muons = np.random.normal(-1, 0.3, 100000)
neutrinos = np.random.normal(1, 0.1, 1000)

plt.hist(muons, 100, color = "red", alpha = 0.5, weights = np.ones(100000) * 1e3, label = "Muons")
plt.hist(neutrinos, 100,  color = "green", alpha = 0.5, weights=np.ones(1000)*1e-3, label = "Neutrinos")
plt.legend()
plt.yscale("log")
plt.ylabel("Amount of events")
plt.xlabel("Optimal criterion")
plt.title("Muon vs. Neutrino proportions")
plt.show()