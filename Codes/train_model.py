from model_training import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import HistGradientBoostingClassifier

muon_folder = "C:\\Users\\lovat\\Desktop\\MyFiles\\Unif\Master's Thesis\\Data\\muon"
data_folder = "C:\\Users\\lovat\\Desktop\\MyFiles\\Unif\Master's Thesis\\Data\\data"
neutrino_folder = "C:\\Users\\lovat\\Desktop\\MyFiles\\Unif\Master's Thesis\\Data\\neutrino"
livetimes_file = "C:\\Users\\lovat\\Desktop\\MyFiles\\Unif\\Master's Thesis\\Data\\livetimes.csv"
save_folder = "C:\\Users\\lovat\\Desktop\\MyFiles\\Unif\Master's Thesis\\Data\\training"
mlp_model = MLPClassifier(solver='adam', alpha=1e-4, hidden_layer_sizes=(10, 10, 10, 5, 5, 2), max_iter=2000)
knn_model = KNeighborsClassifier(n_neighbors=20)
svm_model = SVC(kernel="rbf", gamma="auto", tol=1e-3, cache_size=3*1024, C=1.0)
unbalanced_bdt_model = HistGradientBoostingClassifier(loss='log_loss', max_iter=250, max_leaf_nodes=31, l2_regularization=0.1, max_bins=255 , validation_fraction=0.1, n_iter_no_change=10, tol=1e-07, class_weight=None)
balanced_bdt_model = HistGradientBoostingClassifier(loss='log_loss', max_iter=250, max_leaf_nodes=31, l2_regularization=0.1, max_bins=255 , validation_fraction=0.1, n_iter_no_change=10, tol=1e-07, class_weight="balanced")


#initiateTraining(muon_folder, neutrino_folder, livetimes_file, save_folder,"mlp", mlp_model)
#initiateTraining(muon_folder, neutrino_folder, livetimes_file, save_folder,"knn", knn_model)
#initiateTraining(muon_folder, neutrino_folder, livetimes_file, save_folder,"svm_unbalanced", svm_model, discard=0.95, n_splits=10)
initiateTraining(muon_folder, neutrino_folder, livetimes_file, save_folder,"unbalanced_bdt", unbalanced_bdt_model, discard=0, n_splits=10)
initiateTraining(muon_folder, neutrino_folder, livetimes_file, save_folder,"balanced_bdt", balanced_bdt_model, discard=0, n_splits=10)