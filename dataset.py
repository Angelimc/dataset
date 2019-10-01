import enum
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer as TF
from sklearn.feature_selection import SelectKBest, chi2
import logging
import joblib


logging.basicConfig(level=logging.INFO)
Logger = logging.getLogger('RandomClf.stdout')
Logger.setLevel("INFO")


class Experiment(enum.Enum):
    vt_family_fold = 1
    vt_train_gp_test = 2
    gp_family_fold = 3


def get_feature_filenames(apks, Corpus, extension):
    """ Return list of paths that contain the features for the given apks.
    Parameters
    ----------
    apks : string
        The file path, containing list of apk paths to process.
    Returns
    -------
    samplenames: list
        The list of paths that contain the features.
    """
    samplenames = []
    indexx= 0
    apks_file = open(apks, 'r')
    for apk in apks_file:
        indexx = indexx + 1
        apk = apk.rstrip("\n\r")
        arrays = apk.split("/")
        name = os.path.splitext(arrays[-1])[0]
        absolute_path = os.path.abspath(os.path.join(Corpus, name+ extension))
        if os.path.isfile(absolute_path):
            samplenames.append(absolute_path)
    print(indexx)
    return samplenames


def load_data(model, FeatureVectorizer, x_Btrain_samplenames, x_Mtrain_samplenames, x_Btest_samplenames,
              x_Mtest_samplenames, NumFeaturesToBeSelected):

    # Labels: 0 for benign and 1 for malware
    y_Btrain = np.zeros(len(x_Btrain_samplenames))
    y_Mtrain = np.ones(len(x_Mtrain_samplenames))
    y_Btest = np.zeros(len(x_Btest_samplenames))
    y_Mtest = np.ones(len(x_Mtest_samplenames))

    x_train_samplenames = x_Mtrain_samplenames + x_Btrain_samplenames
    x_train = FeatureVectorizer.fit_transform(x_train_samplenames)
    y_train = np.concatenate((y_Mtrain, y_Btrain), axis=0)

    x_testM = FeatureVectorizer.transform(x_Mtest_samplenames)
    x_testB = FeatureVectorizer.transform(x_Btest_samplenames)

    Features = FeatureVectorizer.get_feature_names()
    Logger.info("Total number of features: {} ".format(len(Features)))

    # FOR CSBD
    if NumFeaturesToBeSelected != -1 and len(Features) > NumFeaturesToBeSelected:
        Logger.info("Gonna select %s features", NumFeaturesToBeSelected)
        FSAlgo = SelectKBest(chi2, k=NumFeaturesToBeSelected)
        x_train = FSAlgo.fit_transform(x_train, y_train)
        x_testM = FSAlgo.transform(x_testM)
        x_testB = FSAlgo.transform(x_testB)

    X = np.concatenate((x_train, x_testM, x_testB), axis=0)
    y = np.concatenate((y_train, y_Mtest, y_Btest), axis=0)

    train_idx = []
    for i in range(len(y_train)):
        train_idx.append(i)

    print("X: ")
    print(X)
    print("y: ")
    print(y)
    print("length train data: ")
    print(str(len(y_train)))
    print("train_idx: ")
    print(train_idx)
    return dict(X=X, y=y, train_idx=train_idx, model=model)


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
    Corpus = '/data/Alex/malware-tools/Drebin/data/'
    model = "/data/Alex/malware-tools/Drebin/GooglePlay/FamilyFold/Fold"+str(fold)+"/modelFoldLinearSVM.pkl"

    FeatureVectorizer = TF(input='filename', tokenizer=lambda x: x.split('\n'), token_pattern=None, binary=True)

    Btrain_file = "/nfs/zeus/michaelcao/DatasetsForTools/Experiments_Updated_August_10/GP_10_Family_Fold_Benign_Train_"\
                  +str(fold)+".txt"
    Mtrain_file = "/nfs/zeus/michaelcao/DatasetsForTools/Experiments_Updated_August_10/GP_10_Family_Fold_Malware_Train_"\
                  +str(fold)+".txt"
    Btest_file = "/nfs/zeus/michaelcao/DatasetsForTools/Experiments_Updated_August_10/GP_10_Family_Fold_Benign_Test_"\
                 +str(fold)+".txt"
    Mtest_file = "/nfs/zeus/michaelcao/DatasetsForTools/Experiments_Updated_August_10/GP_10_Family_Fold_Malware_Test_"\
                 +str(fold)+".txt"

    x_Btrain_samplenames = get_feature_filenames(Btrain_file, Corpus, ".data")
    print(len(x_Btrain_samplenames))
    x_Mtrain_samplenames = get_feature_filenames(Mtrain_file, Corpus, ".data")
    print(len(x_Mtrain_samplenames))
    x_Btest_samplenames = get_feature_filenames(Btest_file, Corpus, ".data")
    print(len(x_Btest_samplenames))
    x_Mtest_samplenames = get_feature_filenames(Mtest_file, Corpus, ".data")
    print(len(x_Mtest_samplenames))

    return load_data(model, FeatureVectorizer, x_Btrain_samplenames, x_Mtrain_samplenames, x_Btest_samplenames,
                     x_Mtest_samplenames, -1)


def MyTokenizer(Str):
    return Str.split()


def load_csbd(fold):
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

    model = "/data/Alex/malware-tools/Drebin/GooglePlay/FamilyFold/Fold"+str(fold)+"/modelFoldLinearSVM.pkl"
    X_filename = "/data/Angeli/visualization_tool/src/csbd_data/X_gp_familyfold_"+str(fold)+".npy"
    y_filename = "/data/Angeli/visualization_tool/src/csbd_data/y_gp_familyfold_"+str(fold)+".npy"
    train_idx_filename = "/data/Angeli/visualization_tool/src/csbd_data/train_idx_gp_familyfold_"+str(fold)+".npy"

    if os.path.isfile(X_filename) and os.path.isfile(y_filename) and os.path.isfile(train_idx_filename):
        return dict(X=np.load(X_filename), y=np.load(y_filename), train_idx=np.load(train_idx_filename), model=model)

    Corpus = '/data/Alex/malware-tools/csbd/data/'

    FeatureVectorizer = TF(input='filename', lowercase=False, token_pattern=None, tokenizer=MyTokenizer,
                           binary=True, dtype=np.float64)

    Btrain_file = "/nfs/zeus/michaelcao/DatasetsForTools/Experiments_Updated_August_10/GP_10_Family_Fold_Benign_Train_"\
                  +str(fold)+".txt"
    Mtrain_file = "/nfs/zeus/michaelcao/DatasetsForTools/Experiments_Updated_August_10/GP_10_Family_Fold_Malware_Train_"\
                  +str(fold)+".txt"
    Btest_file = "/nfs/zeus/michaelcao/DatasetsForTools/Experiments_Updated_August_10/GP_10_Family_Fold_Benign_Test_"\
                 +str(fold)+".txt"
    Mtest_file = "/nfs/zeus/michaelcao/DatasetsForTools/Experiments_Updated_August_10/GP_10_Family_Fold_Malware_Test_"\
                 +str(fold)+".txt"

    x_Btrain_samplenames = get_feature_filenames(Btrain_file, Corpus, ".txt")
    print(len(x_Btrain_samplenames))
    x_Mtrain_samplenames = get_feature_filenames(Mtrain_file, Corpus, ".txt")
    print(len(x_Mtrain_samplenames))
    x_Btest_samplenames = get_feature_filenames(Btest_file, Corpus, ".txt")
    print(len(x_Btest_samplenames))
    x_Mtest_samplenames = get_feature_filenames(Mtest_file, Corpus, ".txt")
    print(len(x_Mtest_samplenames))

    data = load_data(model, FeatureVectorizer, x_Btrain_samplenames, x_Mtrain_samplenames, x_Btest_samplenames,
                     x_Mtest_samplenames, 5000,)

    # Save data to files
    X, y, train_idx = data["X"], data["y"], data["train_idx"]
    np.save(X_filename, X)
    np.save(y_filename, y)
    np.save(train_idx_filename, train_idx)

    return data
