import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from scipy.signal import savgol_filter
from os.path import join as pjoin
from scipy.spatial.distance import jensenshannon
from sklearn.feature_selection import mutual_info_classif


def getOptimalAxis(x, y, w = None):
    columns = x.columns
    res = np.zeros(len(columns))
    if w is None:
        w = np.ones(len(x.index))
    for i in range(len(columns)):
        res[i] = np.sum(w * (x[columns[i]] - x[columns[i]].mean()).to_numpy() * (y - y.mean()).to_numpy())
    return res / np.linalg.norm(res)


def computeOptimalFeature(x, columns, weights):
    res = np.zeros(len(x.index))
    for i in range(len(columns)):
        res += x[columns[i]] * weights[i]
    return res


def displayHistogram(X, feature, nbins=100, useWeights=True):
    if useWeights:
        plt.hist(X[feature][X["type"] == 0], nbins, color = "red", alpha = 0.5, weights=X["weight_bkg"][X["type"] == 0])
        plt.hist(X[feature][X["type"] == 1], nbins,  color = "green", alpha = 0.5, weights=X["weight_sig"][X["type"] == 1])
    else:
        plt.hist(X[feature][X["type"] == 0], nbins, color = "red", alpha = 0.5)
        plt.hist(X[feature][X["type"] == 1], nbins,  color = "green", alpha = 0.5)
    plt.yscale("log")
    plt.show()


def getHistogram(X, weights, types, nbins=100, bins_=None, proba=True, isInt=False):
    Xmax = np.max(X)
    Xmin = np.min(X)
    if bins_ is None:
        if isInt:
            bins = np.arange(Xmin, Xmax+1, 1)
        else:
            bins = np.linspace(Xmin, Xmax, nbins)
    else:
        bins = np.copy(bins_)
    possible_types = []
    for this_type in types:
        if this_type not in possible_types:
            possible_types.append(this_type)
    Xs = [X[types == this_type] for this_type in possible_types]
    ws = [weights[types == this_type] for this_type in possible_types]
    hists = [np.histogram(Xs[i], bins, weights=ws[i], density=proba)[0] for i in range(len(Xs))]
    bins = (bins[:-1] + bins[1:]) / 2

    return hists,  bins


def weightedMean(x, w):
    return np.sum(x * w) / np.sum(w)


def weightedCov(x, y, w):
    return np.sum(w * (x - weightedMean(x, w)) * (y - weightedMean(y, w))) / np.sum(w)


def weightedCorr(x, y, w):
    return weightedCov(x, y, w) / np.sqrt(weightedCov(x, x, w) * weightedCov(y, y, w))


def getMIScores(X, target, n_neighbors=5):
    scores = mutual_info_classif(X, target, n_neighbors=n_neighbors)
    return scores


def getCorrScores(X, target, w):
    scores = []
    for column in X.columns:
        scores.append(weightedCorr(X[column], target, w))
    scores = np.array(scores)
    return scores


def plotHisto(X, Y, w, plot_title, file_name, nbins=100, proba=True, x_axis = "x axis", y_axis = "probability density", x_scale="linear", y_scale="linear"):
    hists, bins = getHistogram(X, w, Y, nbins, proba=proba, isInt=False)
    histMuon = hists[0]
    histNeutrino = hists[1]
    fig, ax = plt.subplots()
    ax.plot(bins, histMuon, color="red")
    ax.fill_between(bins, histMuon, alpha = 0.5, color="red", label="Muons")
    ax.plot(bins, histNeutrino, color="green")
    ax.fill_between(bins, histNeutrino, alpha = 0.5, color="green", label="Neutrinos")

    ax.set_title(plot_title)
    ax.set_xlabel(x_axis)
    ax.set_ylabel(y_axis)
    ax.set_xscale(x_scale)
    ax.set_yscale(y_scale)

    ax.legend()
    plt.savefig(file_name)
    plt.close()


def getJensenShannonScores(X, w, types, plot_folder = None, nbins=100):
    dic = {}
    scores = []
    for feature in X.columns:
        if feature == "cher_nhits" or feature == "n_trig_lines":
            hists, bins = getHistogram(X[feature].to_numpy(), w, types, nbins, proba=True, isInt=True)
        else:
            hists, bins = getHistogram(X[feature].to_numpy(), w, types, nbins, proba=True)
        histMuon = hists[0]
        histNeutrino = hists[1]
        score = jensenshannon(histMuon, histNeutrino)
        scores.append(score)
        if plot_folder is not None:
            fig, ax = plt.subplots()
            ax.plot(bins, histMuon, color="red")
            ax.fill_between(bins, histMuon, alpha = 0.5, color="red", label="Muons")
            ax.plot(bins, histNeutrino, color="green")
            ax.fill_between(bins, histNeutrino, alpha = 0.5, color="green", label="Neutrinos")

            ax.set_title(feature + " - Muon vs NuMu - " + str(round(score, 5)))
            ax.set_xlabel(feature)
            ax.set_ylabel("Probability density")

            ax.legend()
            plot_file = pjoin(plot_folder, feature + ".png")
            plt.savefig(plot_file)
            plt.close()
            dic[feature] = [bins, histMuon, histNeutrino, 1.0, 1.0]
            print(feature + " done")
    scores = np.array(scores)
    return dic, scores


def getProportions(X, w, types, plot_folder, nbins=100, proba=False):
    features = X.columns
    for feature in features:
        fig, ax = plt.subplots()
        if feature == "cher_nhits" or feature == "n_trig_lines":
            hists, bins = getHistogram(X[feature].to_numpy(), w, types, nbins, proba=proba, isInt=True)
        else:
            hists, bins = getHistogram(X[feature].to_numpy(), w, types, nbins, proba=proba)
        histMuon = hists[0]
        histNeutrino = hists[1]
        histTot = histNeutrino + histMuon
        ax.plot(bins, histNeutrino/histTot, color="green")
        ax.fill_between(bins, histNeutrino/histTot, alpha = 0.5, color="green", label="Neutrinos")
        ax.plot(bins, np.ones(len(bins)), color="red")
        ax.fill_between(bins, np.ones(len(bins)), histNeutrino/histTot, alpha = 0.5, color="red", label="Muons")

        ax.set_title(feature + " - Muon vs NuMu")
        ax.set_xlabel(feature)
        ax.set_ylabel("Proportion")
        ax.set_yscale("log")

        ax.legend()
        plot_file = pjoin(plot_folder, feature + ".png")
        plt.savefig(plot_file)
        plt.close()
        print(feature + " done")



def getCorrelationMatrix(X, w, plot_folder=None):
    N = len(X.columns)
    corrMatrix = np.zeros((N, N))
    columns = X.columns
    for i in range(N):
        for j in range(i, N):
            corrMatrix[i, j] = weightedCorr(X[columns[i]].to_numpy(), X[columns[j]].to_numpy(), w)
            corrMatrix[j, i] = corrMatrix[i, j]
    if plot_folder is not None:
        fig, ax = plt.subplots(figsize=(15, 15))
        cax = ax.matshow(corrMatrix, vmin=-1, vmax=1)
        fig.colorbar(cax)
        ticks = np.arange(0,N,1)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        a = 20.0
        ax.set_xticklabels(columns, rotation=90-a, ha='left', va='bottom', fontsize = 8)
        ax.set_yticklabels(columns, rotation=a, ha='right', va='top', fontsize = 8)
        plot_file = pjoin(plot_folder, "Correlations.png")
        plt.savefig(plot_file)
        plt.close()
        fig, ax = plt.subplots(figsize=(15, 15))
        cax = ax.matshow(np.abs(corrMatrix), vmin=0, vmax=1)
        fig.colorbar(cax)
        ticks = np.arange(0,N,1)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        a = 20.0
        ax.set_xticklabels(columns, rotation=90-a, ha='left', va='bottom', fontsize = 8)
        ax.set_yticklabels(columns, rotation=a, ha='right', va='top', fontsize = 8)
        plot_file = pjoin(plot_folder, "Correlations_abs.png")
        plt.savefig(plot_file)
        plt.close()
    return corrMatrix


def interpolateHisto(x, bins, histo):
    new_x = x - bins[0]
    index = new_x // (bins[1] - bins[0])
    c0 = (new_x % (bins[1] - bins[0])) / (bins[1] - bins[0])
    c1 = 1 - c0
    res1 = np.where(x <= bins[0], histo[0], 0)
    res2 = np.where(x >= bins[-1], histo[-1], 0)
    mask = (bins[0] < x) & (x < bins[-1])
    res3 = np.zeros(np.shape(x))
    res3[mask] = histo[index[mask].astype(int)] * c0[mask] + histo[index[mask].astype(int) + 1] * c1[mask]
    res = res1 + res2 + res3
    return res


def computeProbabilityBasedFeature(X, dic, features=None):
    muon_score = np.ones(len(X.index))
    neutrino_score = np.ones(len(X.index))
    if features is None:
        features = dic.keys()
    for feature in features:
        muon_score += dic[feature][3] * interpolateHisto(X[feature].to_numpy(), dic[feature][0], dic[feature][1])
        neutrino_score += dic[feature][4] * interpolateHisto(X[feature].to_numpy(), dic[feature][0], dic[feature][2])
    score = neutrino_score - muon_score
    X["muon_score"] = muon_score
    X["neutrino_score"] = neutrino_score
    X["score"] = score


def getGrad(f, x, d = 1e-5, params = None):
    N = x.shape[-1]
    grad = np.zeros(x.shape)
    x_sup = np.copy(x)
    x_inf = np.copy(x)
    for i in range(N):
        x_sup[i] += d/2
        x_inf[i] -= d/2
        grad[i] = (f(x_sup, params=params) - f(x_inf, params=params)) / d
        x_sup[i] -= d/2
        x_inf[i] += d/2
    return grad



def boundaryMax(f, x0, df = None, max_step = 1000, threshold = 1e-4, dt = 1e-5, params = None):
    delta = 10
    step = 0
    x = np.copy(x0)
    inertia = np.zeros(x.shape)
    maxChange = 0.3
    while step < max_step and delta > threshold:
        if df is not None: 
            gradf = df(x, params=params)
        else:
            gradf = getGrad(f, x, params=params)
        perpGradf = np.dot(x, gradf) * x
        alongGradf = gradf - perpGradf
        effective_inertia = inertia - np.dot(x, inertia) * x
        total_change = alongGradf + effective_inertia
        total_change *= maxChange * (1 / (1 + np.exp(- 2 * np.linalg.norm(total_change) * dt)) - 0.5) / (np.linalg.norm(total_change) * dt)
        inertia = np.copy(total_change)
        delta = np.linalg.norm(total_change)
        x += total_change
        x /= np.linalg.norm(x)
        step += 1
    if(delta <= threshold):
        print(f"threshold reached in {step} steps")
    if(step == max_step):
        print(f"maximal number of steps reached. Last improvement length = {delta}")
    else:
        print(f"final improvement = {delta}")
    return x


def distinguishingDistance(alphas, params):
    df1 = params[0]
    df2 = params[1]
    weights1 = params[2]
    weights2 = params[3]
    columns = df1.columns
    N1 = len(df1.index)
    N2 = len(df1.index)
    meanZ1 = np.sum(alphas * np.array([np.average(df1[column].to_numpy(), weights=weights1) for column in columns]))
    meanZ2 = np.sum(alphas * np.array([np.average(df2[column].to_numpy(), weights=weights2) for column in columns]))
    stdZ1_sqrd = 1 / (N1 -1) * np.sum(weights1 * np.sum(alphas * np.array([df1[column].to_numpy() - np.average(df1[column].to_numpy(), weights=weights1) for column in columns]).T) ** 2)
    stdZ2_sqrd = 1 / (N2 -1) * np.sum(weights2 * np.sum(alphas * np.array([df2[column].to_numpy() - np.average(df2[column].to_numpy(), weights=weights2) for column in columns]).T) ** 2)
    return np.abs(meanZ1 - meanZ2) / (stdZ1_sqrd + stdZ2_sqrd)
    

def selectFeatures(features, scores, correlations, N, correlation_threshold=0.75):
    n = len(features)
    mean_normalized_score = np.zeros(n)
    for score in scores:
        mean_normalized_score += (score - np.min(score)) / np.max(score)
    mean_normalized_score /= scores.shape[0]
    sorted_indices = np.flip(np.argsort(mean_normalized_score))
    selected_indices = []
    i = 0
    while i < n and len(selected_indices) < N:
        if  sorted_indices[i] >= 0:
            selected_indices.append(sorted_indices[i])
            for j in range(n):
                if abs(correlations[sorted_indices[i], sorted_indices[j]]) >= correlation_threshold and sorted_indices[i] != sorted_indices[j]:
                    sorted_indices[j] = -1
        i += 1
    selected_indices = np.array(selected_indices)
    selected_features = features[selected_indices]
    selected_scores = scores[:, selected_indices]
    selected_correlations = correlations[selected_indices][:, selected_indices]
    selected_N = len(selected_indices)
    return selected_features, selected_scores, selected_correlations, selected_N

