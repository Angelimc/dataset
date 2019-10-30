import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer as TF
import logging
from utilities import load_data_gp_familyfold, load_data_vt_familyfold, load_data_gp_test_on_vt


def MyTokenizer(Str):
    return Str.split()


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
    model = "/data/Alex/malware-tools/csbd/GooglePlay/FamilyFold/Old/Fold"+str(fold)+"/modelFold.pkl"
    Corpus = '/data/Alex/malware-tools/csbd/data/'
    FeatureVectorizer = TF(input='filename', lowercase=False, token_pattern=None, tokenizer=MyTokenizer,
                           binary=True, dtype=np.float64)
    data = load_data_gp_familyfold(model, FeatureVectorizer, Corpus, fold, ".txt", 5000)
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


def load_drebin_gp_familyfold(fold):
    model = "/data/Alex/malware-tools/Drebin/GooglePlay/FamilyFold/Fold"+str(fold)+"/modelFoldLinearSVM.pkl"
    Corpus = '/data/Alex/malware-tools/Drebin/data/'
    FeatureVectorizer = TF(input='filename', tokenizer=lambda x: x.split('\n'), token_pattern=None, binary=True)
    return load_data_gp_familyfold(model, FeatureVectorizer, Corpus, fold, ".data", -1)


def load_drebin_gp_test_on_vt():
    model = "/data/Alex/malware-tools/Drebin/GPTestAllVT/modelLinear.pkl"
    Corpus = '/data/Alex/malware-tools/Drebin/data/'
    FeatureVectorizer = TF(input='filename', tokenizer=lambda x: x.split('\n'), token_pattern=None, binary=True)
    return load_data_gp_test_on_vt(model, FeatureVectorizer, Corpus, ".data", -1)


def load_drebin_vt_familyfold(fold):
    model = "/data/Alex/malware-tools/Drebin/VTFamilyFold/Fold"+str(fold+1)+"/modelFoldLinearSVM.pkl"
    Corpus = '/data/Alex/malware-tools/Drebin/data/'
    FeatureVectorizer = TF(input='filename', tokenizer=lambda x: x.split('\n'), token_pattern=None, binary=True)
    return load_data_vt_familyfold(model, FeatureVectorizer, Corpus, fold, ".data", -1)


def select_features_load_drebin_gp_familyfold(fold, stop_words):
    model = "../new_models//Drebin/GooglePlay/Fold"+str(fold)+"/modelFoldLinearSVM.pkl"
    Corpus = '/data/Alex/malware-tools/Drebin/data/'
    FeatureVectorizer = TF(input='filename', tokenizer=lambda x: x.split('\n'), token_pattern=None, binary=True,
                           stop_words=stop_words)
    return load_data_gp_familyfold(model, FeatureVectorizer, Corpus, fold, ".data", -1)


def select_features_alterate_load_drebin_gp_familyfold(fold, min_df):
    model = "../new_models//Drebin/Alternate/GooglePlay/Fold"+str(fold)+"/modelFoldLinearSVM.pkl"
    Corpus = '/data/Alex/malware-tools/Drebin/data/'
    FeatureVectorizer = TF(input='filename', tokenizer=lambda x: x.split('\n'), token_pattern=None, binary=True,
                           min_df=min_df)
    return load_data_gp_familyfold(model, FeatureVectorizer, Corpus, fold, ".data", -1)


def select_optimal_num_features_load_drebin_gp_familyfold(fold):
    model = "../new_models/Drebin/Alternate/GooglePlay/Fold"+str(fold)+"/modelFoldLinearSVM.pkl"
    Corpus = '/data/Alex/malware-tools/Drebin/data/'
    FeatureVectorizer = TF(input='filename', tokenizer=lambda x: x.split('\n'), token_pattern=None, binary=True)
    return load_data_gp_familyfold(model, FeatureVectorizer, Corpus, fold, ".data", -1)

