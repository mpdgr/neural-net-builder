import math
from enum import Enum, auto


class ActivationType(Enum):
    FUNCTION = auto()
    DERIVATIVE = auto()


# relu

def relu(vector, activation):
    if activation == ActivationType.FUNCTION:
        return __relu_vector(vector)
    elif activation == ActivationType.DERIVATIVE:
        return __relu_deriv_vector(vector)
    else:
        print("Activation type not valid")


def __relu_vector(v):
    if type(v) == list and type(v[0]) == list:
        v = v[0]

    for i in range(0, len(v)):
        v[i] = __relu(v[i])
    return v


def __relu_deriv_vector(v):
    if type(v) == list and type(v[0]) == list:
        v = v[0]

    for i in range(0, len(v)):
        v[i] = __relu_deriv(v[i])
    return v


def __relu(x):
    if x > 0:
        return x
    else:
        return 0


def __relu_deriv(x):
    if x > 0:
        return 1
    else:
        return 0


# sigmoid

def sig(vector, activation):
    if activation == ActivationType.FUNCTION:
        return __sig_vector(vector)
    elif activation == ActivationType.DERIVATIVE:
        return __sig_deriv_vector(vector)
    else:
        print("Activation type not valid")


def __sig_vector(v):
    if type(v) == list and type(v[0]) == list:
        v = v[0]

    for i in range(0, len(v)):
        v[i] = __sig(v[i])
    return v


def __sig_deriv_vector(v):
    if type(v) == list and type(v[0]) == list:
        v = v[0]

    for i in range(0, len(v)):
        v[i] = __sig_deriv(v[i])
    return v


def __sig(x):
    return 1 / (1 + math.exp(-x))


def __sig_deriv(x):
    return __sig(x) * (1 - __sig(x))
