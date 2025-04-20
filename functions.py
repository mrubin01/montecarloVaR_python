import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
from scipy.linalg import cholesky


def init_matrix(rows: int, columns: int):
    """
    Initialize a matrix filling it with zeros
    :param rows: number of rows
    :param columns: number of columns
    :return: a matrix with shape rows x columns filled with zeros
    """
    if not isinstance(rows, int) and not isinstance(columns, int):
        return [[]]

    mtrx = np.zeros((rows, columns))

    return mtrx


def init_tensor(days: int, no_of_tickers: int, sim: int):
    """
    Initialize a tensor filling it with zeros
    :param days: nnumber of days
    :param no_of_tickers: number of tickers
    :param sim: number of simulations
    :return: tnsr
    """
    if not isinstance(days, int) and not isinstance(no_of_tickers, int) and not isinstance(sim, int):
        return [[]]

    tnsr = np.zeros((days, no_of_tickers, sim))

    return tnsr


def generate_normal_samples(size: int, mu=0, sigma=1):
    """
    Generate random samples with normal distribution
    :param mu:
    :param sigma:
    :param size:
    :return:
    """
    samples = np.random.normal(mu, sigma, size)

    return samples
