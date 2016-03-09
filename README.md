# multi_instance_models

Multi-instance logistic regression using
[Theano](https://github.com/Theano/Theano) and
[Lasagne](https://github.com/Lasagne/Lasagne).

This includes a class for performing multi-instance logistic
regression using the Theano and Lasagne libraries.  Training instances
are allowed to be grouped into bags, where at least one (but not
necessarily all) instances in a bag are positive.  Negative bags have
all negative instances, and they should be passed as singleton bags to
the training method.

Bag membership of the training instances is given by a vector of bag
labels that must have the same length as the training data.  The
labels must be integers from 0 to num_bags-1, inclusive.  The training
labels must be given for the bags, in array of length num_bags.

There is also an experimental deep network function that is enabled if
the hidden_units parameter is greater than 1.  In this case, a linear
hidden layer with hidden_units hidden units, followed by a maxout
layer with a sigmoid activation function, will be used to compute the
instance-level probabilities.

See the docstrings in the class for more information and all options.

### Installation

Using pip:
```
pip install -r https://raw.githubusercontent.com/matted/multi_instance_models/master/requirements.txt
pip install --upgrade https://github.com/matted/multi_instance_models/archive/master.zip
```

Lasagne requires a version of Theano that is newer than the one
currently on PyPI, which necessitates these extra steps.

You can also look at the [setup_miniconda.sh](setup_miniconda.sh)
script for an example on setting up an isolated Python environment
using [Miniconda](http://conda.pydata.org/miniconda.html).

### Example usage

```
import multiinstance
x = [[1], [2], [3]]
y = [1, 0]
z = [0, 0, 1]
model = multiinstance.MILogisticRegression()
model.fit(x, y, z)
```

See [test.py](test.py) in this repository for a more complete example.

### Notes

* The optimization procedure can be brittle, depending on the
  complexity of the training data and the bagging pattern, so some
  experimentation with the update method and learning rate may be
  required
* The bag labels must be 0 or 1, and this is strictly enforced
* If the bag label codes do not match the number of bag class training
  labels, weird Theano error codes about mismatched dimensions may
  result

### TODO

* Implement the scikit-learn estimator API more carefully so that their
  cross-validation and hyperparameter optimization methods work out of
  the box
* Be more flexible about how bag labels are handled, and do some
  preprocessing or fixing of them if necessary
* Do the same target label preprocessing that scikit-learn does,
  instead of strictly requiring 0 or 1 for all elements
* Factor out the deep network construction so that more
  experimentation can be conducted by the user
* Investigate more complex optimization methods, perhaps via
  scipy.optimize (as in [this
  example](http://deeplearning.net/tutorial/code/logistic_cg.py))
* Implement minibatch training, instead of using the whole dataset
