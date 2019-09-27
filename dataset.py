import enum
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer as TF


class Experiment(enum.Enum):
    vt_family_fold = 1
    vt_train_gp_test = 2
    gp_family_fold = 3


def get_feature_filenames(apks, Corpus):
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
        absolute_path = os.path.abspath(os.path.join(Corpus, name+".data"))
        if os.path.isfile(absolute_path):
            samplenames.append(absolute_path)
    print(indexx)
    return samplenames


def load_data(model, FeatureVectorizer, x_Btrain_samplenames, x_Mtrain_samplenames, x_Btest_samplenames,
              x_Mtest_samplenames):
    y_Btrain = np.zeros(len(x_Btrain_samplenames))
    y_Mtrain = np.ones(len(x_Mtrain_samplenames))
    y_Btest = np.zeros(len(x_Btest_samplenames))
    y_Mtest = np.ones(len(x_Mtest_samplenames))

    x_samplenames = x_Mtrain_samplenames + x_Btrain_samplenames + x_Mtest_samplenames + x_Btest_samplenames
    X = FeatureVectorizer.fit_transform(x_samplenames)
    y = np.concatenate((y_Mtrain, y_Btrain, y_Mtest, y_Btest), axis=0)

    train_idx = []
    for i in range(len(y_Btrain) + len(y_Mtrain)):
        train_idx.append(i)

    print("X: ")
    print(X)
    print("y: ")
    print(y)
    print("length train data: ")
    print(str(len(y_Btrain) + len(y_Mtrain)))
    print("train_idx: ")
    print(train_idx)
    return dict(X=X, y=y, train_idx=train_idx, model=model)


def load_csbd(experiment, fold):
    """Load and return the csbd dataset for given experiment and family fold.
    Parameters
    ----------
    experiment : Experiment, between 1 and 3
        The experiment to load: 1 = Virus total family fold, 2 = Train on virus
        total and test on google play, 3 = Google play family fold
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

    x_Btrain_samplenames = get_feature_filenames(Btrain_file, Corpus)
    print(len(x_Btrain_samplenames))
    x_Mtrain_samplenames = get_feature_filenames(Mtrain_file, Corpus)
    print(len(x_Mtrain_samplenames))
    x_Btest_samplenames = get_feature_filenames(Btest_file, Corpus)
    print(len(x_Btest_samplenames))
    x_Mtest_samplenames = get_feature_filenames(Mtest_file, Corpus)
    print(len(x_Mtest_samplenames))

    return load_data(model, FeatureVectorizer, x_Btrain_samplenames, x_Mtrain_samplenames, x_Btest_samplenames,
                     x_Mtest_samplenames)

