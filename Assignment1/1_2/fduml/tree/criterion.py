""" """ """
criterion
"""

import math


def get_criterion_function(criterion):
    if criterion == "info_gain":
        return __info_gain
    elif criterion == "info_gain_ratio":
        return __info_gain_ratio
    elif criterion == "gini":
        return __gini_index
    elif criterion == "error_rate":
        return __error_rate


def __label_stat(y, l_y, r_y):
    """ Count the number of labels of nodes """
    left_labels = {}
    right_labels = {}
    all_labels = {}
    for t in y.reshape(-1):
        if t not in all_labels:
            all_labels[t] = 0
        all_labels[t] += 1
    for t in l_y.reshape(-1):
        if t not in left_labels:
            left_labels[t] = 0
        left_labels[t] += 1
    for t in r_y.reshape(-1):
        if t not in right_labels:
            right_labels[t] = 0
        right_labels[t] += 1

    return all_labels, left_labels, right_labels


def __info_gain(y, l_y, r_y):
    """
    Calculate the info gain

    y, l_y, r_y: label array of father node, left child node, right child node
    """
    all_labels, left_labels, right_labels = __label_stat(y, l_y, r_y)

    ###########################################################################
    # TODO:                                                                   #
    # Implement this method. Calculate the info gain if splitting y into      #
    # l_y and r_y                                                             #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    if len(l_y) == 0 or len(r_y) == 0:
        return 0

    entropy_before = 0
    for t in all_labels:
        if t > 0:
            entropy_before -= (t/len(y))*math.log2(t/len(y))

    entropy_left = 0
    for t in left_labels:
        if t > 0:
            entropy_left -= (t/len(l_y))*math.log2(t/len(l_y))

    entropy_right = 0
    for t in right_labels:
        if t > 0:
            entropy_right -= (t/len(r_y))*math.log2(t/len(r_y))

    entropy_after = (len(l_y)/len(y))*entropy_left + (len(r_y)/len(y))*entropy_right
    info_gain = entropy_before - entropy_after

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return info_gain


def __info_gain_ratio(y, l_y, r_y):
    """
    Calculate the info gain ratio

    y, l_y, r_y: label array of father node, left child node, right child node
    """
    info_gain = __info_gain(y, l_y, r_y)
    ###########################################################################
    # TODO:                                                                   #
    # Implement this method. Calculate the info gain ratio if splitting y     #
    # into l_y and r_y                                                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    if len(l_y) != 0:
        l = math.log2(len(l_y)/len(y))
    else:
        l = 1 
    if len(r_y) != 0:
        r = math.log2(len(r_y)/len(y))
    else:
        r = 1
    r = len(r_y)/len(y)
    split_info = -(len(l_y)/len(y)*l) - (len(r_y)/len(y)*r)
    info_gain /= split_info
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return info_gain


def __gini_index(y, l_y, r_y):
    """
    Calculate the gini index

    y, l_y, r_y: label array of father node, left child node, right child node
    """
    all_labels, left_labels, right_labels = __label_stat(y, l_y, r_y)

    ###########################################################################
    # TODO:                                                                   #
    # Implement this method. Calculate the gini index value before and        #
    # after splitting y into l_y and r_y                                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    if len(l_y) == 0 or len(r_y) == 0:
        return 0

    before = 1
    for t in all_labels:
        before -= (t/len(y))**2

    gini_l = 1
    for t in left_labels:
        gini_l -= (t/len(l_y))**2

    gini_r = 1
    for t in right_labels:
        gini_r -= (t/len(r_y))**2

    after = (len(l_y)/len(y))*gini_l + (len(r_y)/len(y))*gini_r

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return before - after


def __error_rate(y, l_y, r_y):
    """ Calculate the error rate """
    all_labels, left_labels, right_labels = __label_stat(y, l_y, r_y)

    ###########################################################################
    # TODO:                                                                   #
    # Implement this method. Calculate the error rate value before and        #
    # after splitting y into l_y and r_y                                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    if len(l_y) == 0 or len(r_y) == 0:
        return 0

    before = 1 - (max(all_labels)/len(y))
    after = (len(l_y)/len(y))*(1 - max(left_labels)/len(l_y)) + (len(r_y)/len(y))*(1 - max(right_labels)/len(r_y))

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return before - after
