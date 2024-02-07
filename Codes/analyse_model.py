from utils import *
from model_training import *
from model_analysis import *


muon_folder = "C:\\Users\\lovat\\Desktop\\MyFiles\\Unif\Master's Thesis\\Data\\muon"
data_folder = "C:\\Users\\lovat\\Desktop\\MyFiles\\Unif\Master's Thesis\\Data\\data"
neutrino_folder = "C:\\Users\\lovat\\Desktop\\MyFiles\\Unif\Master's Thesis\\Data\\neutrino"
livetimes_file = "C:\\Users\\lovat\\Desktop\\MyFiles\\Unif\\Master's Thesis\\Data\\livetimes.csv"
save_folder = "C:\\Users\\lovat\\Desktop\\MyFiles\\Unif\Master's Thesis\\Data\\training"
plot_folder = "C:\\Users\\lovat\\Desktop\\MyFiles\\Unif\Master's Thesis\\Plots\\models2"
keep = 0.1
run_names = ["mlp", "svm_unbalanced"] # , "knn", "svm_unbalanced", "unbalanced_bdt", "balanced_bdt"
for run_name in run_names:
    analyseModel(muon_folder, data_folder, neutrino_folder, livetimes_file, save_folder, plot_folder, run_name, keep=1.0, save_name=run_name+"_scores", nbins=100)



