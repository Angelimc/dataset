import enum
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer as TF
from sklearn.feature_selection import SelectKBest, chi2
import logging
from joblib import dump, load
from scipy.sparse import coo_matrix, vstack
from tempfile import TemporaryFile
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV



logging.basicConfig(level=logging.INFO)
Logger = logging.getLogger('RandomClf.stdout')
Logger.setLevel("INFO")


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
        absolute_path = os.path.abspath(os.path.join(Corpus, name + extension))
        if os.path.isfile(absolute_path):
            samplenames.append(absolute_path)
    print(indexx)
    return samplenames


def load_data(model, FeatureVectorizer, x_Btrain_samplenames, x_Mtrain_samplenames, x_Btest_samplenames,
              x_Mtest_samplenames, NumFeaturesToBeSelected):

    # Labels: 0 for benign and 1 for malware
    y_Btrain = np.zeros(len(x_Btrain_samplenames), dtype=int)
    y_Mtrain = np.ones(len(x_Mtrain_samplenames), dtype=int)
    y_Btest = np.zeros(len(x_Btest_samplenames), dtype=int)
    y_Mtest = np.ones(len(x_Mtest_samplenames), dtype=int)

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

    x_train = coo_matrix(x_train)
    x_testM = coo_matrix(x_testM)
    x_testB = coo_matrix(x_testB)
    X = vstack([x_train, x_testM, x_testB]).toarray()
    y = np.concatenate((y_train, y_Mtest, y_Btest), axis=0)

    train_idx = []
    for i in range(len(y_train)):
        train_idx.append(i)

    """
    Parameters = {'n_estimators': [10,50,100,200,500,1000], 'bootstrap': [True, False],'criterion': ['gini', 'entropy']}
    Clf = GridSearchCV(RandomForestClassifier(), Parameters, cv=5, scoring='f1', n_jobs=3)
    RFmodels = Clf.fit(x_train, y_train)
    BestModel = RFmodels.best_estimator_
    """
    RFmodels = load(model)
    print("malware predictions:")
    y_Mpred = RFmodels.predict(x_testM)
    print(y_Mpred)
    print("benign predictions:")
    y_Bpred = RFmodels.predict(x_testB)
    print(y_Bpred)

    print("X: ")
    print(X)
    print("y: ")
    print(y)
    print("length train data: ")
    print(str(len(y_train)))
    print("train_idx: ")
    print(train_idx)
    return dict(X=X, y=y, train_idx=train_idx, model=model)


def load_data_gp_familyfold(model, FeatureVectorizer, Corpus, fold, extension, NumFeaturesToBeSelected):
    Btrain_file = "/nfs/zeus/michaelcao/DatasetsForTools/Experiments_Updated_August_10/GP_10_Family_Fold_Benign_Train_"\
                  +str(fold)+".txt"
    Mtrain_file = "/nfs/zeus/michaelcao/DatasetsForTools/Experiments_Updated_August_10/GP_10_Family_Fold_Malware_Train_"\
                  +str(fold)+".txt"
    Btest_file = "/nfs/zeus/michaelcao/DatasetsForTools/Experiments_Updated_August_10/GP_10_Family_Fold_Benign_Test_"\
                 +str(fold)+".txt"
    Mtest_file = "/nfs/zeus/michaelcao/DatasetsForTools/Experiments_Updated_August_10/GP_10_Family_Fold_Malware_Test_"\
                 +str(fold)+".txt"

    x_Btrain_samplenames = get_feature_filenames(Btrain_file, Corpus, extension)
    print(len(x_Btrain_samplenames))
    x_Mtrain_samplenames = get_feature_filenames(Mtrain_file, Corpus, extension)
    print(len(x_Mtrain_samplenames))
    x_Btest_samplenames = get_feature_filenames(Btest_file, Corpus, extension)
    print(len(x_Btest_samplenames))
    x_Mtest_samplenames = get_feature_filenames(Mtest_file, Corpus, extension)
    print(len(x_Mtest_samplenames))

    return load_data(model, FeatureVectorizer, x_Btrain_samplenames, x_Mtrain_samplenames, x_Btest_samplenames,
                     x_Mtest_samplenames, NumFeaturesToBeSelected)


# RandomClassificationFamilyVt.py
def load_data_vt_familyfold(model, FeatureVectorizer, Corpus, fold, extension, NumFeaturesToBeSelected):
    Btrain_file = "/nfs/zeus/michaelcao/DatasetsForTools/Experiments_Updated_August_18/VT_10_Family_Fold_Benign_Train_"\
                  +str(fold)+".txt"
    Mtrain_file = "/nfs/zeus/michaelcao/DatasetsForTools/Experiments_Updated_August_18/VT_10_Family_Fold_Malware_Train_"\
                  +str(fold)+".txt"
    Btest_file = "/nfs/zeus/michaelcao/DatasetsForTools/Experiments_Updated_August_18/VT_10_Family_Fold_Benign_Test_"\
                 +str(fold)+".txt"
    Mtest_file = "/nfs/zeus/michaelcao/DatasetsForTools/Experiments_Updated_August_18/VT_10_Family_Fold_Malware_Test_"\
                 +str(fold)+".txt"

    x_Btrain_samplenames = get_feature_filenames(Btrain_file, Corpus, extension)
    print(len(x_Btrain_samplenames))
    x_Mtrain_samplenames = get_feature_filenames(Mtrain_file, Corpus, extension)
    print(len(x_Mtrain_samplenames))
    x_Btest_samplenames = get_feature_filenames(Btest_file, Corpus, extension)
    print(len(x_Btest_samplenames))
    x_Mtest_samplenames = get_feature_filenames(Mtest_file, Corpus, extension)
    print(len(x_Mtest_samplenames))

    return load_data(model, FeatureVectorizer, x_Btrain_samplenames, x_Mtrain_samplenames, x_Btest_samplenames,
                     x_Mtest_samplenames, NumFeaturesToBeSelected)


def load_data_gp_test_on_vt(model, FeatureVectorizer, Corpus, extension, NumFeaturesToBeSelected):
    Btrain_file = "/nfs/zeus/michaelcao/DatasetsForTools/Experiments_Updated_August_18/All_Benign_Samples.txt"
    Mtrain_file = "/nfs/zeus/michaelcao/DatasetsForTools/Experiments_Updated_August_18/All_VT_Malware_Samples.txt"
    Mtest_file_2016 = "/nfs/zeus/michaelcao/DatasetsForTools/Experiments_Updated_August_18/GP_Malware_2016.txt"
    Mtest_file_2017 = "/nfs/zeus/michaelcao/DatasetsForTools/Experiments_Updated_August_18/GP_Malware_2017.txt"
    Mtest_file_2018 = "/nfs/zeus/michaelcao/DatasetsForTools/Experiments_Updated_August_18/GP_Malware_2018.txt"
    Mtest_file_2019 = "/nfs/zeus/michaelcao/DatasetsForTools/Experiments_Updated_August_18/GP_Malware_2019.txt"

    x_Btrain_samplenames = get_feature_filenames(Btrain_file, Corpus, extension)
    print(len(x_Btrain_samplenames))
    x_Mtrain_samplenames = get_feature_filenames(Mtrain_file, Corpus, extension)
    print(len(x_Mtrain_samplenames))
    x_Btest_samplenames = []
    x_Mtest_samplenames = get_feature_filenames(Mtest_file_2016, Corpus, extension)
    x_Mtest_samplenames += get_feature_filenames(Mtest_file_2017, Corpus, extension)
    x_Mtest_samplenames += get_feature_filenames(Mtest_file_2018, Corpus, extension)
    x_Mtest_samplenames += get_feature_filenames(Mtest_file_2019, Corpus, extension)
    print(len(x_Mtest_samplenames))

    return load_data(model, FeatureVectorizer, x_Btrain_samplenames, x_Mtrain_samplenames, x_Btest_samplenames,
                     x_Mtest_samplenames, NumFeaturesToBeSelected)
