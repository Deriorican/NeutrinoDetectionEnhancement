import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from os.path import join as pjoin
from os.path import isfile
from os import listdir
from utils import *
from scipy import stats
from sklearn.model_selection import KFold
import pickle


def setupDataframes(X_train, X_test, Y_train, Y_test, W_train, W_test):
    ind = (np.abs(stats.zscore(X_train)) < 3).all(axis=1)
    X_train_clean = X_train[ind]
    Y_train_clean = Y_train[ind]
    W_train_clean = W_train[ind]

    X_test_clean = (X_test-X_train_clean.mean())/X_train_clean.std()
    X_train_clean = (X_train_clean-X_train_clean.mean())/X_train_clean.std()
    return X_train_clean, X_test_clean, Y_train_clean, Y_test, W_train_clean, W_test



def trainModel(X_train, X_test, Y_train, Y_test, W_train, W_test, model, computeScore=False):
    X_train_clean, X_test_clean, Y_train_clean, Y_test, W_train_clean, W_test = \
        setupDataframes(X_train, X_test, Y_train, Y_test, W_train, W_test)
    _, JSscores = getJensenShannonScores(X_train_clean, W_train_clean, Y_train_clean.to_numpy())
    MIscores = getMIScores(X_train_clean, Y_train_clean, 5)
    corrMatrix = getCorrelationMatrix(X_train_clean, W_train_clean)

    features = X_train_clean.columns.to_numpy()
    selected_features, selected_scores, selected_correlations, selected_N = \
        selectFeatures(features, np.array([JSscores, MIscores]), corrMatrix, 10, correlation_threshold=0.6)
    X_train_selected = X_train_clean[selected_features]
    X_test_selected = X_test_clean[selected_features]
    model.fit(X_train_selected, Y_train_clean)
    if computeScore:
        Y_predicted = model.predict(X_test_selected)
        score = model.score(X_test_selected, Y_test)
    else:
        Y_predicted = None
        score = None
    return model, Y_predicted, score, selected_features



def savePKL(to_save, file):
    with open(file, 'wb') as handle:
        pickle.dump(to_save, handle, protocol=pickle.HIGHEST_PROTOCOL)



def readPKL(file):
    with open(file, 'rb') as handle:
        return pickle.load(handle)



def getData(muon_folder, data_folder, neutrino_folder, livetimes_file):
    dataframes = []
    livetimes = pd.read_csv(livetimes_file)
    for filename in listdir(muon_folder):
        splitted = filename.split("Run")
        run = int(splitted[-1][:-3])
        file = pjoin(muon_folder, filename)
        dataframes.append(pd.read_hdf(file))
        livetime = livetimes["Livetime"].loc[livetimes["Run"] == run].values[0]
        dataframes[-1]["weight_bkg"] /= livetime
        dataframes[-1]["weight_sig"] /= livetime
    X_muon_raw = pd.concat(dataframes)

    dataframes = []
    for filename in listdir(neutrino_folder):
        if filename[:4] == "numu":
            splitted = filename.split("Run")
            run = int(splitted[-1][:-3])
            file = pjoin(neutrino_folder, filename)
            dataframes.append(pd.read_hdf(file))
            livetime = livetimes["Livetime"].loc[livetimes["Run"] == run].values[0]
            dataframes[-1]["weight_bkg"] /= livetime
            dataframes[-1]["weight_sig"] /= livetime
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
    X_raw = pd.concat(dataframes)

    X_raw = X_raw[:][X_raw["n_trig_hits"] > 20]
    X_raw = X_raw[:][X_raw["n_trig_lines"] >= 2]
    X_raw = X_raw[:][X_raw["bestmuon_dz"] < 0]

    print(f"Dataframe composed of {len(X_raw.index)} samples")
    return X_raw



def prepareData(X_raw, train_index, test_index):
    X_train = X_raw.iloc[train_index]
    X_test = X_raw.iloc[test_index]

    W_train = np.where(X_train["type"] == 0, X_train["weight_bkg"], X_train["weight_sig"])
    W_test = np.where(X_test["type"] == 0, X_test["weight_bkg"], X_test["weight_sig"])

    Y_train = X_train["type"]
    Y_test = X_test["type"]
    
    X_test = X_test.drop(columns=["weight_sig", "weight_bkg", "run", "type"])
    X_train = X_train.drop(columns=["weight_sig", "weight_bkg", "run", "type"])
    X_train["track_energy"] = np.log(X_train["track_energy"].to_numpy() + 1e-15)
    X_test["track_energy"] = np.log(X_test["track_energy"].to_numpy() + 1e-15)
    return X_train, X_test, Y_train, Y_test, W_train, W_test



def initiateTraining(muon_folder, data_folder, neutrino_folder, livetimes_file, save_folder, run_name, model, discard=0.0, n_splits=10, computeScore=False):

    print("building dataframe...")

    X_raw = getData(muon_folder, data_folder, neutrino_folder, livetimes_file)
    
    print("\ndataframe built...\n")

    kf = KFold(n_splits=n_splits, shuffle=True)

    models = []
    if computeScore:
        scores = []
    train_indices = []
    test_indices = []

    for i, (train_index_raw, test_index_raw) in enumerate(kf.split(X_raw)):
        n_train_raw = len(train_index_raw)
        discarded_train = np.random.choice(train_index_raw, size = int(discard * n_train_raw), replace=False)
        test_index = np.append(test_index_raw, discarded_train)
        train_index = np.setdiff1d(train_index_raw, discarded_train)
        print(f"Fold {i} : {len(train_index)} train samples; {len(test_index)} test samples.")
        
        X_train, X_test, Y_train, Y_test, W_train, W_test = prepareData(X_raw, train_index, test_index)

        model, _, score, _= trainModel(X_train, X_test, Y_train, Y_test, W_train, W_test, model, computeScore=computeScore)
        models.append(model)
        train_indices.append(train_index)
        test_indices.append(test_index)
        if computeScore:
            scores.append(score)
            print(f"\t score = {score}\n")
    
    if computeScore:
        print(f"mean score on all folds = {np.mean(scores)}\n")

    models_file = pjoin(save_folder, run_name + "_models.pickle")
    train_indices_file = pjoin(save_folder, run_name + "_train.pickle")
    test_indices_file = pjoin(save_folder, run_name + "_test.pickle")
    savePKL(models, models_file)
    savePKL(train_indices, train_indices_file)
    savePKL(test_indices, test_indices_file)
