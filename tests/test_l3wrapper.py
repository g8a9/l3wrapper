import pytest
from l3wrapper.l3wrapper import L3Classifier
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import os
from joblib import dump, load
import pickle


@pytest.fixture
def dataset_X_y():
    X = np.loadtxt('tests/data/car.data', dtype=object, delimiter=',')
    y = X[:, -1]
    X = X[:, :-1]
    return X, y


def test_fit_predict(dataset_X_y):
    X, y = dataset_X_y
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    clf = L3Classifier().fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    assert y_pred.shape[0] == X_test.shape[0]
    assert len(clf.labeled_transactions_) == X_test.shape[0]
    print(clf.labeled_transactions_[1].matched_rules,
            clf.labeled_transactions_[1].used_level)
    print(len([t for t in clf.labeled_transactions_ if t.used_level == -1]))


def test_save_human_readable(dataset_X_y):
    X, y = dataset_X_y
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    clf = L3Classifier().fit(X_train, y_train, save_human_readable=True)
    train_dir = clf.current_token_
    files = [f for f in os.listdir(train_dir) if f.startswith(f"{clf.current_token_}")]
    assert len(files) == 2 # level 1 and level 2


def test_training_files(dataset_X_y):
    X, y = dataset_X_y
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    clf = L3Classifier().fit(X_train, y_train, remove_files=False)
    train_dir = clf.current_token_
    files = [f for f in os.listdir(train_dir) if f.startswith(f"{clf.current_token_}")]
    assert len(files) == 7 # all the stuff left by L3 


def test_save_load(dataset_X_y):
    X, y = dataset_X_y
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    clf = L3Classifier().fit(X_train, y_train)

    # test dump with joblib pre-predict
    dump(clf, "clf_pre_predict.joblib")
    y_pred = clf.predict(X_test)
    assert y_pred.shape[0] == X_test.shape[0]
    assert len(clf.labeled_transactions_) == X_test.shape[0]

    # test dump with joblib post-predict
    dump(clf, "clf.joblib")
    clf_l = load("clf.joblib")
    assert len(clf.lvl1_rules_) == len(clf_l.lvl1_rules_)

    # test dump with pickle
    with open("clf.pickle", "wb") as fp:
        pickle.dump(clf, fp)
    with open("clf.pickle", "rb") as fp:
        clf_l = pickle.load(fp)
    assert len(clf.lvl2_rules_) == len(clf_l.lvl2_rules_)


@pytest.fixture
def get_column_names():
    return ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']

# 
# X = np.loadtxt('car.data', dtype=object, delimiter=',')
# y = X[:, -1]
# X = X[:, :-1]


@pytest.fixture
def get_param_grid():
    return {
        'min_sup': [0.1, 0.01],
        'min_conf': [0.5, 0.75],
        'max_matching': [1, 2, 5],
        'specialistic_rules': [True, False],
        'max_length': [0, 2]
    }


def test_grid_search(dataset_X_y, get_param_grid):
    X, y = dataset_X_y
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    clf = GridSearchCV(L3Classifier(), get_param_grid, n_jobs=-1)
    clf.fit(X, y)

    print(clf.best_estimator_)
    print(clf.best_score_)


def test_leve1_modifier(dataset_X_y):
    X, y = dataset_X_y
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    clf = L3Classifier(rule_sets_modifier='level1').fit(X, y)
    assert clf.n_lvl2_rules_ == 0 and len(clf.lvl2_rules_) == 0

    y_pred = clf.predict(X_test)
    assert y_pred.shape[0] == X_test.shape[0]
    assert len(clf.labeled_transactions_) == X_test.shape[0]
    print(clf.labeled_transactions_[1].matched_rules,
            clf.labeled_transactions_[1].used_level)
    assert len([
        t for t in clf.labeled_transactions_ if t.used_level == 2
    ]) == 0
