"""
metrics of classification
"""

import numpy as np


def accuracy_score(y_true, y_pred):
    """Accuracy classification score.
    In multilabel classification, this function computes subset accuracy:
    the set of labels predicted for a sample must *exactly* match the
    corresponding set of labels in y_true.
    Read more in the :ref:`User Guide <accuracy_score>`.

    Parameters
    ----------
    y_true : 1d array-like, or label indicator array / sparse matrix
        Ground truth (correct) labels.
    y_pred : 1d array-like, or label indicator array / sparse matrix
        Predicted labels, as returned by a classifier.

    Returns
    -------
    score : accuracy
    """
    acc = -1
    #########################################################################
    # TODO:                                                                 #
    # Calculate the accuracy.                                               #
    #########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    equal = (y_true == y_pred)
    acc = np.sum(equal) / equal.shape[0]
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return acc
