import enum
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer as TF
from sklearn.feature_selection import SelectKBest, chi2
import logging
import joblib
from scipy.sparse import coo_matrix, vstack
from tempfile import TemporaryFile
from utilities import load_data_gp_familyfold, load_data_vt_familyfold


class Experiment(enum.Enum):
    vt_family_fold = 1
    vt_train_gp_test = 2
    gp_family_fold = 3


def MyTokenizer(Str):
    return Str.split()


def load_drebin(fold):
    """Load and return the Drebin dataset for Google play family fold experiment and family fold.
    Parameters
    ----------
    fold : integer, between 0 and 9
        The family fold to load.
    Returns
    -------
    data : dictionary
        Dictionary, the attributes are:
        X : array, shape = [n_samples, n_features]
            A {n_samples by n_samples} size matrix containing data
        y : array, shape = [n_samples]
            Labels (0 = benign, 1 = malware)
        training_indices : array,
            Indices on which the classifier has been trained
        model: string
            filepath containing classification model
    """
    model = "/data/Alex/malware-tools/Drebin/GooglePlay/FamilyFold/Fold"+str(fold)+"/modelFoldLinearSVM.pkl"
    Corpus = '/data/Alex/malware-tools/Drebin/data/'
    FeatureVectorizer = TF(input='filename', tokenizer=lambda x: x.split('\n'), token_pattern=None, binary=True)
    return load_data_gp_familyfold(model, FeatureVectorizer, Corpus, fold, ".data", -1)


def load_csbd_gp_familyfold(fold):
    """Load and return the CSBD dataset for Google play family fold experiment and family fold.
    Parameters
    ----------
    fold : integer, between 0 and 9
        The family fold to load.
    Returns
    -------
    data : dictionary
        Dictionary, the attributes are:
        X : array, shape = [n_samples, n_features]
            A {n_samples by n_samples} size matrix containing data
        y : array, shape = [n_samples]
            Labels (0 = benign, 1 = malware)
        training_indices : array,
            Indices on which the classifier has been trained
        model: string
            filepath containing classification model
    """

    model = "/data/Alex/malware-tools/csbd/GooglePlay/FamilyFold/Fold"+str(fold)+"/modelFold.pkl"
    X_filename = "/data/Angeli/visualization_tool/src/csbd_data/X_gp_familyfold_"+str(fold)+".npy"
    y_filename = "/data/Angeli/visualization_tool/src/csbd_data/y_gp_familyfold_"+str(fold)+".npy"
    train_idx_filename = "/data/Angeli/visualization_tool/src/csbd_data/train_idx_gp_familyfold_"+str(fold)+".npy"

#    if os.path.isfile(X_filename) and os.path.isfile(y_filename) and os.path.isfile(train_idx_filename):
#        return dict(X=np.load(X_filename), y=np.load(y_filename), train_idx=np.load(train_idx_filename), model=model)

    Corpus = '/data/Alex/malware-tools/csbd/data/'
    FeatureVectorizer = TF(input='filename', lowercase=False, token_pattern=None, tokenizer=MyTokenizer,
                           binary=True, dtype=np.float64)
    data = load_data_gp_familyfold(model, FeatureVectorizer, Corpus, fold, ".txt", 5000)

    # Save data to files
#    X, y, train_idx = data["X"], data["y"], data["train_idx"]
#    np.save(X_filename, X)
#    np.save(y_filename, y)
#    np.save(train_idx_filename, train_idx)
    return data


def load_csbd_vt_familyfold(fold):
    model = "/data/Alex/malware-tools/csbd/VTFamilyFold/Fold"+str(fold+1)+"/model.pkl"
    Corpus = '/data/Alex/malware-tools/csbd/data/'
    FeatureVectorizer = TF(input='filename', lowercase=False, token_pattern=None, tokenizer=MyTokenizer,
                           binary=True, dtype=np.float64)
    return load_data_vt_familyfold(model, FeatureVectorizer, Corpus, fold, ".txt", 5000)


def load_csbd_gp_test_on_vt():
    model = "/data/Alex/malware-tools/csbd/GPTestOnVTAll/model.pkl"
    Corpus = '/data/Alex/malware-tools/csbd/data/'
    FeatureVectorizer = TF(input='filename', lowercase=False, token_pattern=None, tokenizer=MyTokenizer,
                           binary=True, dtype=np.float64)
    return load_data_gp_test_on_vt(model, FeatureVectorizer, Corpus, ".txt", 5000)
