import os
import sys
import numpy as np


def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)


def moving_average(a, n=3):
    if len(a) < n + 1:
        n = len(a) - 1
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def moving_avarage_smoothing(X, k=3):
    """
    Moving average of a multivariate time series.
    """
    S = np.zeros(X.shape)
    for t in range(X.shape[0]):
        if t < k:
            S[t] = np.mean(X[:t+1], axis=0)
        else:
            S[t] = np.sum(X[t-k:t], axis=0)/k
    return S


def unison_shuffled_copies(a):
    if type(a) == list:
        if len(a) == 3:
            assert len(a[0]) == len(a[1]) == len(a[2])
            p = np.random.permutation(len(a[0]))
            return [a[0][p], a[1][p], a[2][p]]
        elif len(a) == 2:
            assert len(a[0]) == len(a[1])
            p = np.random.permutation(len(a[0]))
            return [a[0][p], a[1][p]]
        else:
            print("!!! no unison shuffle possible, input list has the wrong size !!!")
            raise NotImplementedError
    else:
        print("!!! no unison shuffle possible, input must be list !!!")
        raise TypeError


def soft_thresh(x, l):
    """
    shrink function
    """
    return np.sign(x) * np.maximum(np.abs(x) - l, 0.)


def debugger_is_on():
    return sys.gettrace() is not None


def tfcall(f):
    return f if debugger_is_on() else getattr(f.__self__, '_'+f.__name__)


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

    # Method that returns a message with the desired color
    # usage:
    #    print(bcolor.colored("My colored message", bcolor.OKBLUE))
    @staticmethod
    def colored(message, color):
        return color + message + bcolors.ENDC

    # Method that returns a yellow warning
    # usage:
    #   print(bcolors.warning("What you are about to do is potentially dangerous. Continue?"))
    @staticmethod
    def warning(message):
        return bcolors.WARNING + message + bcolors.ENDC

    # Method that returns a red fail
    # usage:
    #   print(bcolors.fail("What you did just failed massively. Bummer"))
    #   or:
    #   sys.exit(bcolors.fail("Not a valid date"))
    @staticmethod
    def fail(message):
        return bcolors.FAIL + message + bcolors.ENDC

    # Method that returns a green ok
    # usage:
    #   print(bcolors.ok("What you did just ok-ed massively. Yay!"))
    @staticmethod
    def ok(message):
        return bcolors.OKGREEN + message + bcolors.ENDC

    # Method that returns a blue ok
    # usage:
    #   print(bcolors.okblue("What you did just ok-ed into the blue. Wow!"))
    @staticmethod
    def okblue(message):
        return bcolors.OKBLUE + message + bcolors.ENDC

    # Method that returns a header in some purple-ish color
    # usage:
    #   print(bcolors.header("This is great"))
    @staticmethod
    def header(message):
      return bcolors.HEADER + message + bcolors.ENDC


def printok(message):
    print(bcolors.ok(message))


def printokblue(message):
    print(bcolors.okblue(message))


def printwarning(message):
    print(bcolors.warning(message))


def printerror(message):
    print(bcolors.fail(message))


def printfail(message):
    print(bcolors.fail(message))


def find_segments(data: np.ndarray):
    changes = np.diff(data, prepend=0, append=0)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1) - 1
    segments = np.column_stack((starts, ends)).tolist()
    return segments


def get_value_from_tensor(tensor):
    return tensor.to('cpu').detach().numpy()