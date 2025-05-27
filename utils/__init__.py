# Utility functions for the project.

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import ipdb


def create_if_not_exists(path: str) -> None:
    """
    Creates the specified folder if it does not exist.
    Args:
        -path: the complete path of the folder to be created.
    """
    if not os.path.exists(path):
        os.makedirs(path)


def valid_loss(f):
    """
    Decorator function that prevents training to nan loss value.
    """

    def decorated_f(*args, **kwargs):
        loss = f(*args, **kwargs)
        if torch.isnan(loss):
            ipdb.set_trace()
            loss = f(*args, **kwargs)
        return loss

    return decorated_f


def timer(func):
    """
    Decorator function that prints the execution time of the decorated function.
    """

    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(
            f"Function '{func.__name__}' took {execution_time:.6f} seconds to execute."
        )
        return result

    return wrapper


def is_module_differentiable(module):
    """
    Checks if a module is differentiable.
    """
    parameters = list(module.parameters())
    buffers = list(module.buffers())

    if len(parameters) == 0 and len(buffers) == 0:
        # If the module has no parameters or buffers, it is not differentiable
        return False

    # Check if all the parameters and buffers are differentiable
    for param in parameters + buffers:
        if not param.requires_grad:
            return False

    return True
