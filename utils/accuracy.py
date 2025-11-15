def accuracy(df, col):
    """
    Computes the average error of the naive model
    :param df: pandas df
    :param col: the column to compute the accuracy
    :return: (float) mean absolute error between real and expected value
    """
    return (df["HS_after_gapfill"] - df[col]).abs().mean()