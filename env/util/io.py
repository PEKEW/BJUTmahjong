"""Utility functions for Input/Output."""
import argparse
import os
import torch
from torch.optim import Adam


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


class NoDataRootError(Exception):
    """Exception to be thrown when data root doesn't exist."""
    pass


def get_data_root():
    data_root_var = 'DATA_SET'
    try:
        return os.environ[data_root_var]
    except KeyError:
        raise NoDataRootError('Data root must be in environment variable {}, which'
                              ' doesn\'t exist.'.format(data_root_var))


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace