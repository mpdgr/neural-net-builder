import math
import random
from util.matrix_util import *


# --------------------------- apply dropout to output vector ---------------------------
def apply_dropout(dropout_rate, v):
    if len(v) <= 1:
        print("Output vector must consist of at least 2 elements")
        return v
    if dropout_rate >= 1 or dropout_rate < 0:
        print("Dropout rate must be in range <0, 1)")
        return v

    if dropout_rate == 0:
        return v

    # nr of elements to drop
    drop = math.floor(len(v) * dropout_rate)
    if len(v) - drop < 2:
        print("Too high dropout rate. Dropout not applied")
        return v

    # dropout rate adjusted to real nr of elements
    actual_dropout_rate = drop / len(v)
    print(actual_dropout_rate)

    drop_indices = random.sample(range(len(v)), drop)

    print(drop_indices)

    dropout_vector = v.copy()


    for index in drop_indices:
        dropout_vector[index] = 0

    print(dropout_vector)

    # multiply vector elements (effectively remaining elements by actual dropout p

    adjusted_dropout_vector = vector_scalar_product(dropout_vector, 1 / (1 - actual_dropout_rate))
    print(adjusted_dropout_vector)

    return adjusted_dropout_vector




