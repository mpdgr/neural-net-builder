from enum import Enum, auto

from exception.InvalidArgumentException import InvalidArgumentException


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
        raise InvalidArgumentException("Loss type not valid!")


def __mse_vector(target_v, actual_v):
    target_v, actual_v = __type_check(target_v, actual_v)
    if len(target_v) != len(actual_v):
        raise InvalidArgumentException("Target and output vectors must be same length!")

    return [__mse(t, a) for t, a in zip(target_v, actual_v)]


def __mse_deriv_vector(target_v, actual_v):
    target_v, actual_v = __type_check(target_v, actual_v)
    if len(target_v) != len(actual_v):
        raise InvalidArgumentException("Target and output vectors must be same length!")

    return [__mse_deriv(t, a) for t, a in zip(target_v, actual_v)]


def __mse(target, actual):
    return 0.5 * (target - actual) ** 2


def __mse_deriv(target, actual):
    return actual - target


def __type_check(*v):
    checked = []
    for v in v:
        if type(v) == list and type(v[0]) == list:
            checked.append(v[0])
        elif type(v) == int:
            checked.append([v])
        else:
            checked.append(v)
    return checked

