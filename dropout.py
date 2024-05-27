import logging as log
import math
import random
from util.matrix_util import *


# --------------------------- apply dropout to output vector ---------------------------
def apply_dropout(dropout_rate, v):
    # returns tuple: (input vector after dropout, list indicating at which indices dropout was applied)

    if len(v) <= 1 and dropout_rate != 0:
        log.warning("Output vector must consist of at least 2 elements. Dropout not applied!")
        return v, []
    if dropout_rate >= 1 or dropout_rate < 0:
        log.warning("Dropout rate must be in range <0, 1). Dropout not applied!")
        return v, []

    if dropout_rate == 0:
        return v, []

    # nr of elements to drop
    drop = math.floor(len(v) * dropout_rate)
    if len(v) - drop < 2:
        log.warning("Too high dropout rate. Dropout not applied!")
        return v, []

    # dropout rate adjusted to actual nr of elements
    actual_dropout_rate = drop / len(v)

    # select random indices to drop
    drop_indices = random.sample(range(len(v)), drop)

    # copy of input vector to apply dropout
    dropout_vector = v.copy()

    # replace values at given indices in the vector with zeros
    for index in drop_indices:
        dropout_vector[index] = 0

    # multiply vector elements (effectively remaining elements by actual dropout rate to compensate
    # for dropout in total layer output
    adjusted_dropout_vector = vector_scalar_product(dropout_vector, 1 / (1 - actual_dropout_rate))
    log.debug(f'Adjusted dropout vector: {adjusted_dropout_vector}')

    return adjusted_dropout_vector, drop_indices


# --------------------------- apply dropout in backpropagation -------------------------
def apply_backprop_dropout(drop_indices, v):
    # zero gradients for dropped out neurons
    for i in drop_indices:
        v[i] = 0

    # verify actual dropout rate
    dropout_rate = len(drop_indices) / len(v)

    # increase remaining gradients according to dropout rate
    return [x if dropout_rate == 0 else x / dropout_rate for x in v]

