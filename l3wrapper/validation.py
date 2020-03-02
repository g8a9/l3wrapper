def check_column_names(X, column_names):
    # TODO handle the case where X is a column vector
    if len(column_names) != X.shape[1]:
        raise ValueError("The number of column names and columns in X are different.")