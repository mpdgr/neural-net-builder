import math
import copy
from enum import Enum, auto


class LossType(Enum):
    FUNCTION = auto()
    DERIVATIVE = auto()


# mean square error

def mse(target_v, actual_v, loss):
    if loss == LossType.FUNCTION:
        return __mse_vector(target_v, actual_v)
    elif loss == LossType.DERIVATIVE:
        return __mse_deriv_vector(target_v, actual_v)
    else:
        print("Loss type not valid")


def __mse_vector(target_v, actual_v):
    if type(target_v) == list and type(target_v[0]) == list:
        target_v = target_v[0]
    if type(actual_v) == list and type(actual_v[0]) == list:
        actual_v = actual_v[0]

    if type(target_v) == int:
        target_v = [target_v]
    if type(actual_v) == int:
        actual_v = [actual_v]

    if len(target_v) != len(actual_v):
        "Target and output vectors must be same length"
        return

    return [__mse(t, a) for t, a in zip(target_v, actual_v)]


def __mse_deriv_vector(target_v, actual_v):
    # todo: parse extract
    if type(target_v) == list and type(target_v[0]) == list:
        target_v = target_v[0]
    if type(actual_v) == list and type(actual_v[0]) == list:
        actual_v = actual_v[0]

    if type(target_v) == int:
        target_v = [target_v]
    if type(actual_v) == int:
        actual_v = [actual_v]

    if len(target_v) != len(actual_v):
        "Target and output vectors must be same length"
        return

    return [__mse_deriv(t, a) for t, a in zip(target_v, actual_v)]


def __mse(target, actual):
    return 0.5 * (target - actual) ** 2


def __mse_deriv(target, actual):
    return actual - target


