import multiinstance
import numpy
import sklearn.cross_validation
import sklearn.datasets
import sklearn.metrics
import sklearn.linear_model
import lasagne.updates

mi_lr = multiinstance.MILogisticRegression(max_iter=200, 
                                           penalty=None)

deep_mi_lr = multiinstance.MILogisticRegression(max_iter=200,
                                                hidden_units=5,
                                                penalty=None)

sparse_mi_lr = multiinstance.MILogisticRegression(max_iter=200, 
                                                  penalty="l1", 
                                                  C=0.025)

lr = sklearn.linear_model.LogisticRegression(max_iter=200, 
                                             # so that penalty (1/C) is small:
                                             C=1e6)

sparse_lr = sklearn.linear_model.LogisticRegression(max_iter=200, 
                                                    penalty="l1", 
                                                    C=0.025)

X, y = sklearn.datasets.make_classification(n_samples=1024*4,
                                            n_features=20,
                                            n_informative=10,
                                            weights=(0.5, 0.5),
                                            n_classes=2)

X_train, X_test, y_train, y_test = sklearn.cross_validation.train_test_split(X, y)

# print X_train.shape, y_train.shape, X_test.shape, y_test.shape

for label, model, train_args in [("unbagged MI-LR", mi_lr, (X_train, y_train)),
                                 ("unbagged sparse MI-LR", sparse_mi_lr, (X_train, y_train)),
                                 ("bagged MI-LR", mi_lr, (X_train, y_train, numpy.arange(len(y_train)))),
                                 ("bagged sparse MI-LR", sparse_mi_lr, (X_train, y_train, numpy.arange(len(y_train)))),
                                 ("scikit-learn LR", lr, (X_train, y_train)),
                                 ("scikit-learn sparse LR", sparse_lr, (X_train, y_train)),
                                 ("deep unbagged MI-LR", deep_mi_lr, (X_train, y_train)),
]:
    print label
    model.fit(*train_args)

    print "\ttraining iterations:", model.n_iter_
    print "\tnonzero feature weights:", len(numpy.nonzero(model.coef_)[0])
    print "\t'significant' feature weights:", numpy.sum(numpy.abs(model.coef_.flatten()) > 1e-2)
    # print "\tweights:", model.coef_.flatten()

    # stupid trick to handle predictions the same way (MI-LR is an array of length N, LR is 2D, size N*2)
    print "\ttraining AUC:", sklearn.metrics.roc_auc_score(y_train, model.predict_log_proba(X_train).T.flatten()[-len(y_train):])
    print "\ttesting AUC:", sklearn.metrics.roc_auc_score(y_test, model.predict_log_proba(X_test).T.flatten()[-len(y_test):])
