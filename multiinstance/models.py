import numpy
import lasagne, lasagne.updates, lasagne.regularization
from lasagne import layers
import theano
from theano import tensor as T

class MILogisticRegression(object):
    """Multi-instance logistic regression classifier.

    A class for performing [deep] multi-instance logistic regression
    using the Theano and Lasagne libraries.  Training instances are
    allowed to be grouped into bags, where at least one (but not
    necessarily all) instances in a bag are positive.  Negative bags
    have all negative instances, and they should be passed as
    singleton bags to the training method.

    If hidden_units is greater than 1, a linear hidden layer with that
    many hidden units, followed by a maxout layer with a sigmoid
    activation function, will be used to compute the instance-level
    probabilities.

    Parameters
    ----------
    fit_intercept : whether to include an intercept term in the model
    max_iter : maximum number of iterations for the fitting algorithm
    tol : minimum change in loss function to determine convergence
    penalty : regularization penalty for the weights, l1, l2, or None
    hidden_units : number of hidden units followed by a maxout layer
    updater : update function lasagne.updates to optimize the loss
    C : inverse regularization penalty (like scikit-learn)
    learning_rate : learning rate of the update algorithm

    """
    def __init__(self, fit_intercept=True, max_iter=100, tol=1e-5,
                 penalty="l2", hidden_units=1,
                 updater=lasagne.updates.nesterov_momentum, C=1.0,
                 learning_rate=0.1):
        
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.updater = updater
        self.learning_rate = learning_rate
        self.C = C
        self.tol = tol
        self.hidden_units = hidden_units
        
        if penalty == "l2":
            self.penalty = lasagne.regularization.l2
        elif penalty == "l1":
            self.penalty = lasagne.regularization.l1
        elif penalty is None:
            self.C = None
            self.penalty = None
        else:
            raise ValueError("penalty must be 'l1', 'l2', or None")
            
    def _setup_model(self, num_features, num_rows):
        if self.fit_intercept:
            b = lasagne.init.Constant(0.)
        else:
            b = None
            
        X_sym = T.matrix()
        y_sym = T.ivector()
        bag_labels = T.ivector()
            
        input_layer = layers.InputLayer(shape=(num_rows, num_features),
                                        input_var=X_sym)
        
        if self.hidden_units <= 1:
            instance_log_odds = layers.DenseLayer(input_layer,
                                                  num_units=1,
                                                  W=lasagne.init.Constant(0.),
                                                  b=b,
                                                  nonlinearity=lasagne.nonlinearities.linear)
        else:
            instance_log_odds = layers.DenseLayer(input_layer,
                                                  num_units=self.hidden_units,
                                                  W=lasagne.init.GlorotUniform(1.0),
                                                  b=b,
                                                  nonlinearity=lasagne.nonlinearities.linear)

            instance_log_odds = layers.FeaturePoolLayer(instance_log_odds,
                                                        pool_size=self.hidden_units,
                                                        pool_function=T.max)
            
        instance_log_odds = layers.FlattenLayer(instance_log_odds, outdim=1)

        instance_log_odds_output = layers.get_output(instance_log_odds, X_sym)
        instance_probs_output = T.nnet.sigmoid(instance_log_odds_output)

        self.all_params = layers.get_all_params(instance_log_odds, trainable=True)
        
        bag_mapper = T.transpose(T.extra_ops.to_one_hot(bag_labels, T.max(bag_labels)+1))
        # if previous layers were probabilities:
        # bag_probs = 1 - T.exp(T.dot(bag_mapper, T.log(1 - instance_probs_output)))
        # if previous layers were log odds:
        bag_probs = 1 - T.exp(T.dot(bag_mapper, -T.nnet.softplus(instance_log_odds_output))) 
        
        if self.C is None:
            regularization = 0
        else:
            # I scale the penalty by num_rows since the likelihood
            # term is the average over instances, instead of the sum
            # (like sklearn).  This is to make the learning rate not
            # depend on the dataset (or minibatch) size, but it means
            # we have to know the minibatch size here in order for C
            # to be the same as for sklearn.
            #
            # Note: this applies the same regularization to all
            # "regularizable" parameters in the whole network
            # (everything but the bias terms).  I need to think more
            # about whether this makes sense for the deep networks,
            # though it's probably a reasonable starting point.
            regularization = 1.0/self.C/num_rows * lasagne.regularization.regularize_network_params(instance_log_odds, self.penalty)
        
        # This chunk is a bit repetitive and could be simplified:
        bag_loss = T.mean(lasagne.objectives.binary_crossentropy(bag_probs, y_sym)) + regularization
        self.f_train_bag = theano.function([X_sym, y_sym, bag_labels], 
                                           [bag_loss],
                                           updates=self.updater(bag_loss,
                                                                self.all_params,
                                                                learning_rate=self.learning_rate))
            
        nobag_loss = T.mean(lasagne.objectives.binary_crossentropy(instance_probs_output, y_sym)) + regularization
        self.f_train_nobag = theano.function([X_sym, y_sym], 
                                             [nobag_loss], 
                                             updates=self.updater(nobag_loss,
                                                                  self.all_params,
                                                                  learning_rate=self.learning_rate))
                
        self.f_bag_logprobs = theano.function([X_sym, bag_labels], T.log(bag_probs))
        self.f_logprobs = theano.function([X_sym], T.log(instance_probs_output))
    
    def fit(self, X_train, y_train, bag_labels=None):
        """Fit the model according to the given training data.

        Parameters
        ----------
        X_train : array-like, shape (n_samples, n_features)
            Training vector, where n_samples in the number of samples and
            n_features is the number of features.
        y_train : array-like, shape (n_bags)
            Bag-level target vector relative to X_train and
            bag_labels.  All labels must be 0 or 1, and there must be
            samples of each type given.
        bag_labels : array-like, shape (n_samples) optional
            Bag labels for each training instance (rows of X_train),
            which must be integers from 0 to n_bags-1.  If no labels
            are given, the model falls bag to standard logistic
            regression (each instance will have a bag of size 1, and
            class labels are no longer noisy).  Negative bags should
            all be size 1, since the model does not handle them
            correctly otherwise.
        Returns
        -------
        self : object
            Returns self.

        """
        if len(numpy.unique(y_train)) != 2 or numpy.min(y_train) < 0.0 or numpy.max(y_train) > 1.0:
            raise ValueError("class labels should all be 0 or 1, and both should be present")

        X_train = numpy.asarray(X_train, dtype=numpy.float32)
        y_train = numpy.asarray(y_train, dtype=numpy.int32)
        
        # Always set up a new model, since it depends on the input
        # size and that may have changed since the last call (if any).
        self._setup_model(X_train.shape[1], X_train.shape[0])
            
        train_args = [X_train, y_train]
        
        if bag_labels is None:
            # Train with a simpler objective if instances aren't bagged.
            train = self.f_train_nobag
        else:
            bag_labels = numpy.asarray(bag_labels, dtype=numpy.int32)
            train = self.f_train_bag
            train_args.append(bag_labels)
         
        last = numpy.inf
        for epochs in xrange(self.max_iter):
            curr = train(*train_args)[0]
            if numpy.abs(last - curr) < self.tol:
                break
            last = curr
            
        self.n_iter_ = epochs
        
        if self.fit_intercept:
            self.intercept_ = self.all_params[1].get_value()
        else:
            self.intercept_ = 0.
        self.coef_ = self.all_params[0].get_value()

        return self
        
    def predict_log_proba(self, X_test):
        """Log of instance-level probability estimates.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
        Returns
        -------
        T : array-like, shape = [n_samples]
            Returns the log-probability of each sample for being class 1.
        """
        return self.f_logprobs(numpy.asarray(X_test, dtype=numpy.float32))

    def predict(self, X_test):
        """Instance-level class predictions.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]

        Returns
        -------
        T : array-like, shape = [n_samples]
            Returns the predicted class of each sample (0 or 1).
        """
        return numpy.asarray(self.predict_log_proba(X_test) >= numpy.log(0.5), dtype=numpy.int32)
    
    def predict_proba(self, X_test):
        """Instance-level probability estimates.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
        Returns
        -------
        T : array-like, shape = [n_samples]
            Returns the probability of each sample for being class 1.
        """
        return numpy.exp(self.predict_log_proba(X_test))
    
    def predict_log_bag_proba(self, X_test, bag_labels):
        """Log of bag-level probability estimates.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
        bag_labels : array-like, shape = [n_samples, n_bags]
            A list of bag labels for the samples, required to be
            integers from 0 to n_bags-1.

        Returns
        -------
        T : array-like, shape = [n_bags]
            Returns the log-probability of each bag for being class 1.
        """
        return self.f_bag_logprobs(numpy.asarray(X_test, dtype=numpy.float32),
                                   numpy.asarray(bag_labels, dtype=numpy.int32))
    
    def predict_bag_proba(self, X_test, bag_labels):
        """Bag-level probability estimates.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
        bag_labels : array-like, shape = [n_samples, n_bags]
            A list of bag labels for the samples, required to be
            integers from 0 to n_bags-1.

        Returns
        -------
        T : array-like, shape = [n_bags]
            Returns the probability of each bag for being class 1.
        """
        return numpy.exp(self.predict_log_bag_proba(X_test, bag_labels))
        
    def predict_bag(self, X_test, bag_labels):
        """Bag-level class predictions.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
        bag_labels : array-like, shape = [n_samples, n_bags]
            A list of bag labels for the samples, required to be
            integers from 0 to n_bags-1.

        Returns
        -------
        T : array-like, shape = [n_bags]
            Returns the predicted class of each bag (0 or 1).
        """
        return numpy.asarray(self.predict_log_bag_proba(X_test, bag_labels) >= numpy.log(0.5), dtype=numpy.int32)
