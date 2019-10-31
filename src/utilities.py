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
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV, Lasso
import joblib
from sklearn.metrics import mean_squared_error


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


def feature_selection_by_lasso_num(x_train, y_train, x_testM, x_testB, x_Btest_samplenames, n):
    """ Use Lasso to select top n features
    Parameters
    ----------
    x_train: X data that will be reduced to n features.
    y_train: y data that will be reduced to n features.
    """
    Logger.info("Gonna select top {} features".format(n))
    # We use the base estimator LassoCV
    clf = LassoCV(cv=5)
    # Set a max number of features to select from
    sfm = SelectFromModel(clf, max_features=n)
    x_train = sfm.fit_transform(x_train, y_train)
    x_testM = sfm.transform(x_testM)
    if len(x_Btest_samplenames) > 0:
        x_testB = sfm.transform(x_testB)
    Logger.info("Selected {} features".format(x_train.shape[1]))
    assert (x_train.shape[1] == n and x_testM.shape[1] == n)
    return x_train, x_testM, x_testB


def feature_selection_by_lasso_alpha(X_train, y_train, X_test, y_test, Features):
    """ Use Lasso to select features
    """
    Logger.info("Total features: {}".format(X_train.shape[1]))

    #lassocv = LassoCV(alphas=None, cv=10, max_iter=100000, normalize=True)
    #lassocv.fit(X_train, y_train)

    lasso = Lasso(max_iter=10000, normalize=True)
    #lasso.set_params(alpha=lassocv.alpha_)
    lassocv_calculate_alpha = 0.00026253495223639133
    lasso.set_params(alpha=lassocv_calculate_alpha)
    # lasso.fit(X_train, y_train)

    # For Lasso, the threshold defaults to 1e-5
    sfm = SelectFromModel(lasso)
    X_train = sfm.fit_transform(X_train, y_train)
    X_test = sfm.transform(X_test)

    mse = mean_squared_error(y_test, sfm.estimator_.predict(X_test))
    train_score = sfm.estimator_.score(X_train, y_train)
    test_score = sfm.estimator_.score(X_test, y_test)
    coeff_used = np.sum(sfm.estimator_.coef_ != 0)
    print("alpha selected: " + str(sfm.estimator_.alpha_))
    print("mean squared error: " + str(mse))
    print("training score: " + str(train_score))
    print("test score: " + str(test_score))
    print("number of features used: " + str(coeff_used))

    mask = sfm.get_support() #list of booleans
    new_features = []
    for bool, feature in zip(mask, Features):
        if bool:
            new_features.append(feature)
    Logger.info("selected {} features".format(len(new_features)))
    assert(len(new_features) == X_train.shape[1] and X_train.shape[1] == X_test.shape[1])

    return X_train, X_test, new_features

    # lasso = Lasso()
    # lasso.fit(X_train, y_train)
    # train_score = lasso.score(X_train, y_train)
    # test_score = lasso.score(X_test, y_test)
    # coeff_used = np.sum(lasso.coef_ != 0)
    # print("training score: " + str(train_score))
    # print("test score: " + str(test_score))
    # print("number of features used: " + str(coeff_used))
    #
    # lasso001 = Lasso(alpha=0.01, max_iter=10e5)
    # lasso001.fit(X_train, y_train)
    # train_score001 = lasso001.score(X_train, y_train)
    # test_score001 = lasso001.score(X_test, y_test)
    # coeff_used001 = np.sum(lasso001.coef_ != 0)
    # print("training score for alpha=0.01: " + str(train_score001))
    # print("test score for alpha =0.01: " + str(test_score001))
    # print("number of features used: for alpha =0.01: " + str(coeff_used001))
    #
    # lasso00001 = Lasso(alpha=0.0001, max_iter=10e5)
    # lasso00001.fit(X_train, y_train)
    # train_score00001 = lasso00001.score(X_train, y_train)
    # test_score00001 = lasso00001.score(X_test, y_test)
    # coeff_used00001 = np.sum(lasso00001.coef_ != 0)
    # print("training score for alpha=0.0001: " + str(train_score00001))
    # print("test score for alpha =0.0001: " + str(test_score00001))
    # print("number of features used: for alpha =0.0001:" + str(coeff_used00001))
    #
    # lasso000001 = Lasso(alpha=0.00001, max_iter=10e5)
    # lasso000001.fit(X_train, y_train)
    # train_score000001 = lasso000001.score(X_train, y_train)
    # test_score000001 = lasso000001.score(X_test, y_test)
    # coeff_used000001 = np.sum(lasso000001.coef_ != 0)
    # print("training score for alpha=0.00001: " + str(train_score000001))
    # print("test score for alpha =0.00001: " + str(test_score000001))
    # print("number of features used: for alpha =0.00001:" + str(coeff_used000001))

    # Features = Features[(sfm.get_support())]
    # Logger.info("Selected features: {}".format(X_train.shape[1]))
    # assert(len(Features) == len(X_train.shape[1]))


def feature_selection_by_SelectKBest(x_train, y_train, x_testM, x_testB, x_Btest_samplenames, n):
    """ Use SelectKBest to select top n features
    Parameters
    ----------
    x_train: X data that will be reduced to n features.
    y_train: y data that will be reduced to n features.
    """
    Logger.info("Gonna select %s features", n)
    FSAlgo = SelectKBest(chi2, k=n)
    x_train = FSAlgo.fit_transform(x_train, y_train)
    x_testM = FSAlgo.transform(x_testM)
    if len(x_Btest_samplenames) > 0:
        x_testB = FSAlgo.transform(x_testB)
    Logger.info("Selected {} features".format(x_train.shape[1]))
    return x_train, x_testM, x_testB


def load_data(model, FeatureVectorizer, X_Btrain_samplenames, X_Mtrain_samplenames, X_Btest_samplenames,
              X_Mtest_samplenames, NumFeaturesToBeSelected):
    # Create labels: 0 for benign and 1 for malware
    y_Btrain = np.zeros(len(X_Btrain_samplenames), dtype=int)
    y_Mtrain = np.ones(len(X_Mtrain_samplenames), dtype=int)
    y_Btest = np.zeros(len(X_Btest_samplenames), dtype=int)
    y_Mtest = np.ones(len(X_Mtest_samplenames), dtype=int)
    y_train = np.concatenate((y_Mtrain, y_Btrain), axis=0)
    y_test = np.concatenate((y_Mtest, y_Btest), axis=0)

    # Extract features
    X_train_samplenames = X_Mtrain_samplenames + X_Btrain_samplenames
    X_train = FeatureVectorizer.fit_transform(X_train_samplenames)
    X_test_samplenames = X_Mtest_samplenames + X_Btest_samplenames
    X_test = FeatureVectorizer.transform(X_test_samplenames)
    assert(len(y_train) == X_train.shape[0] and len(y_test) == X_test.shape[0])

    # Create new model (ie. with feature selection) if file path contains new_models
    if "new_models" in model:
        if not os.path.isfile(model):
            Logger.info("Perform Classification with SVM Model")
            Parameters = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
            ClfLinear = GridSearchCV(LinearSVC(), Parameters, cv=5, scoring='f1', n_jobs=10)
            SVMLinearModels = ClfLinear.fit(X_train, y_train)
            dump(ClfLinear, model)
        else:
            ClfLinear = joblib.load(model)
        SVMLinearModels = ClfLinear.fit(X_train, y_train)
        train_score = SVMLinearModels.score(X_train, y_train)
        test_score = SVMLinearModels.score(X_test, y_test)
        print("linear svc training score: " + str(train_score))
        print("linear svc test score: " + str(test_score))

    # Get feature names
    Features = FeatureVectorizer.get_feature_names()
    Logger.info("Total number of features: {} ".format(len(Features)))

    # For drebin
    #X_train, x_testM, x_testB, Features = feature_selection_by_lasso_alpha(X_train, y_train, X_test, y_test, Features)
    X_train, X_test, Features = feature_selection_by_lasso_alpha(X_train, y_train, X_test, y_test, Features)

    print(X_train.shape[1])
    print(X_test.shape[1])
    print("X_train")
    print(X_train)
    print("x_testM")
    print(X_test)
    print("y_train")
    print(y_train)

    # FOR CSBD
    if NumFeaturesToBeSelected != -1 and len(Features) > NumFeaturesToBeSelected:
        Logger.info("Gonna select %s features", NumFeaturesToBeSelected)
        FSAlgo = SelectKBest(chi2, k=NumFeaturesToBeSelected)
        X_train = FSAlgo.fit_transform(X_train, y_train)
        X_test = FSAlgo.transform(X_test)

    # Combine train and test data in a matrix, to create X (shape = [n_samples, n_features])
    X_train = coo_matrix(X_train)
    X_test = coo_matrix(X_test)
    X = vstack([X_train, X_test]).toarray()
    y = np.concatenate((y_train, y_test), axis=0)

    train_idx = []
    for i in range(len(y_train)):
        train_idx.append(i)

    print("X: ")
    print(X)
    print("y: ")
    print(y)
    print("length train data: ")
    print(str(len(y_train)))
    print("length train_idx: ")
    print(str(len(train_idx)))
    print("length of features:")
    print(str(len(Features))) # NOTE: I did not filter features for CSBD (top 5000)
    return dict(X=X, y=y, train_idx=train_idx, model=model, features=Features)


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
