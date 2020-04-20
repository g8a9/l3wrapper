"""
This module provides several validation functions used by the estimator.
"""

import numpy as np


_NON_CATEGORICAL_ERROR = RuntimeError("L3 cannot handle numeric inputs. Use only 'object' or 'StringDtype'" +
                                        " if you are using pandas.")


def check_column_names(X, column_names):
    """Check the column names specified by the user.

    By design, the character ':' is not allowed in any column name.
    """
    # TODO handle the case where X is a column vector
    if len(column_names) != X.shape[1]:
        raise ValueError("The number of column names and columns in X are different.")

    for name in column_names:
        if ":" in name:
            raise ValueError("The character ':' is not allowed in column names.")


def check_dtype(array):
    """Check the type of input values given by the user.

    No subclasses on :class:`numpy.number` are allowed."""

    # pandas dataframe
    if hasattr(array, "dtypes") and hasattr(array.dtypes, '__array__'):
        for dtype in list(array.dtypes):
            if np.issubdtype(dtype, np.number):
                raise _NON_CATEGORICAL_ERROR
        return array.values
    elif hasattr(array, "dtype"):
        if np.issubdtype(array.dtype, np.number):
            raise _NON_CATEGORICAL_ERROR
        return array
    else:
        raise RuntimeError("Cannot identify 'dtype' in the input. Use only numpy arrays or pandas dataframes.")
