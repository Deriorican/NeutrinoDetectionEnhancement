from utils import *
from model_training import *


def loadModelResults(save_folder, run_name):
    models_file = pjoin(save_folder, run_name + "_models.pickle")
    train_indices_file = pjoin(save_folder, run_name + "_train.pickle")
    test_indices_file = pjoin(save_folder, run_name + "_test.pickle")
    models = readPKL(models_file)
    train_indices = readPKL(train_indices_file)
    test_indices = readPKL(test_indices_file)
    return models, train_indices, test_indices



def computeScores(Y_predicted, Y_true):
    n_muons = (Y_true==0).sum()
    n_neutrinos = (Y_true==1).sum()
    true_muons = np.logical_and(Y_predicted==0, Y_true==0).sum()
    true_neutrinos = np.logical_and(Y_predicted==1, Y_true==1).sum()
    false_muons = np.logical_and(Y_predicted==0, Y_true==1).sum()
    false_neutrinos = np.logical_and(Y_predicted==1, Y_true==0).sum()
    accuracy = (true_muons + true_neutrinos) / (true_neutrinos + true_muons + false_muons + false_neutrinos)
    mu_precision = true_muons / (true_muons + false_muons)
    nu_precision = true_neutrinos / (true_neutrinos + false_neutrinos)
    mu_recall = true_muons / (true_muons + false_neutrinos)
    nu_recall = true_neutrinos / (true_neutrinos + false_muons)
    return n_muons, n_neutrinos, true_muons, true_neutrinos, false_muons, false_neutrinos,\
           accuracy, mu_precision, nu_precision, mu_recall, nu_recall



def analyseModel(muon_folder, neutrino_folder, livetimes_file, save_folder, plot_folder, run_name, keep=1.0, save_name=None, nbins=100):
    models, train_indices, test_indices = loadModelResults(save_folder, run_name)
    X_raw = getData(muon_folder, neutrino_folder, livetimes_file)
    for i in range(len(models)):
        print("===================================================================")
        print(f"Fold {i}:")
        model = models[i]
        train_index = train_indices[i]
        test_index = np.random.choice(test_indices[i], size = int(keep * len(test_indices[i])), replace=False)
        print(len(test_index))
        X_train, X_test, Y_train, Y_test, W_train, W_test = \
            prepareData(X_raw, train_index, test_index)
        X_train_clean, X_test_clean, Y_train_clean, Y_test, W_train_clean, W_test = \
            setupDataframes(X_train, X_test, Y_train, Y_test, W_train, W_test)

        X_train_selected = X_train_clean[model.feature_names_in_]
        X_test_selected = X_test_clean[model.feature_names_in_]
        Y_predicted = model.predict(X_test_selected)
        n_muons, n_neutrinos, true_muons, true_neutrinos, false_muons, false_neutrinos,\
           accuracy, mu_precision, nu_precision, mu_recall, nu_recall = computeScores(Y_predicted, Y_test)
        print(f"\t {n_muons} total muons")
        print(f"\t {n_neutrinos} total neutrions")
        print("-------------------------------------------------------------------")
        print(f"\t {true_muons} true muons")
        print(f"\t {true_neutrinos} true neutrinos")
        print(f"\t {false_muons} false muons")
        print(f"\t {false_neutrinos} false neutrinos")
        print("-------------------------------------------------------------------")
        print(f"\t accuracy = {accuracy}")
        print(f"\t neutrinos precision = {nu_precision}")
        print(f"\t muons precision = {mu_precision}")
        print(f"\t neutrinos recall = {nu_recall}")
        print(f"\t muons recall = {mu_recall}")
        print("===================================================================")
        print("\n\n")
        result_array = np.array([n_muons, n_neutrinos, true_muons, true_neutrinos, false_muons,\
                                 false_neutrinos, accuracy, nu_precision, mu_precision, nu_recall, mu_recall])
        predicted_proba = model.predict_proba(X_test_selected)
        
        plot_title = "Model results distribution"
        feature_file_name = pjoin(plot_folder, run_name + f"_{i}.png")
        save_file = pjoin(save_folder, save_name + ".csv")
        if save_name is not None:
            np.savetxt(save_file, result_array)
        plotHisto(predicted_proba[:, 1], Y_test, W_test, plot_title, feature_file_name, nbins=nbins, proba=True, x_axis = "result's feature")
        
        steps = np.linspace(0, 1, nbins)
        roc = []
        nu_precisions = []
        nu_recalls = []
        hists, bins = getHistogram(predicted_proba[:, 1], W_test, Y_test.values, bins_=steps, proba=True, isInt=False)
        histMuon = hists[0]
        histNeutrino = hists[1]
        muon_kept = np.cumsum(histMuon)
        neutrino_kept = np.cumsum(histNeutrino)
        for j in range(nbins):
            threshold = steps[j]
            Y_predicted = np.where(predicted_proba[:, 1] >= threshold, 1, 0)
            n_muons, n_neutrinos, true_muons, true_neutrinos, false_muons, false_neutrinos,\
                accuracy, mu_precision, nu_precision, mu_recall, nu_recall = computeScores(Y_predicted, Y_test)
            if ((true_neutrinos + false_muons) != 0) and (true_muons + false_neutrinos != 0):
                roc.append([true_neutrinos / (true_neutrinos + false_muons), false_neutrinos / (true_muons + false_neutrinos)])
            elif ((true_neutrinos + false_muons) != 0) and (true_muons + false_neutrinos == 0):
                roc.append([true_neutrinos / (true_neutrinos + false_muons), 0])
            elif ((true_neutrinos + false_muons) == 0) and (true_muons + false_neutrinos != 0):
                roc.append([0, false_neutrinos / (true_muons + false_neutrinos)])
            else:
                roc.append([0, 0])
            nu_precisions.append(nu_precision)
            nu_recalls.append(nu_recall)
        roc = np.array(roc)
        nu_precisions = np.array(nu_precisions)
        nu_recalls = np.array(nu_recalls)
        
        graph_file_name0 = pjoin(plot_folder, run_name + f"_{i}_analysis0.png")
        fig, axs = plt.subplots(1, 2, figsize=(18, 8))
        axs[0].plot(roc[:, 1], roc[:, 0], color="green", label = "ROC")
        axs[0].plot(roc[:, 1], roc[:, 1], color="gray", linestyle="--")
        axs[0].set_title("ROC curve")
        axs[0].set_xlabel("False neutrino rate")
        axs[0].set_ylabel("True neutrino rate")
        axs[0].legend()
        axs[0].grid(True, linestyle="dotted")

        axs[1].plot(steps, nu_precisions, color="green", label = "neutrino precision")
        axs[1].plot(steps, nu_recalls, color="orange", label = "neutrino recall")
        axs[1].set_title("Precision vs Recall")
        axs[1].set_xlabel("threshold on model's probabilities")
        axs[1].legend()
        axs[1].grid(True, linestyle="dotted")
        plt.savefig(graph_file_name0)
        plt.close()

        graph_file_name1 = pjoin(plot_folder, run_name + f"_{i}_analysis1.png")
        fig2, ax = plt.subplots(1, 1)
        ax.plot(bins, 100 -neutrino_kept, color="green", label = "% neutrinos kept")
        ax.plot(bins, 100 - muon_kept, color="red", label = "% muons kept")
        ax.set_title("Particles kept after threshold")
        ax.set_xlabel("threshold on model's probabilities")
        ax.set_ylabel("Particles kept [%]")
        ax.legend()
        ax.grid(True, linestyle="dotted")
        plt.savefig(graph_file_name1)
        plt.close()


