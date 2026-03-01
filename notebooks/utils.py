import numpy as np 

ci_to_quantiles = {
    '95': [0.025, 0.975],
    '90': [0.05, 0.95],
    '50': [0.25, 0.75],
}

epsilon = 1e-3

def filter_outliers(df, column_name):
    """
    Filter out outliers from a DataFrame based on the IQR method.
    
    Parameters:
    df (pd.DataFrame): The DataFrame to filter.
    column_name (str): The name of the column to filter.
    
    Returns:
    pd.DataFrame: The filtered DataFrame.
    """
    Q1 = df[column_name].quantile(0.25)
    Q3 = df[column_name].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column_name] >= lower_bound) & (df[column_name] <= upper_bound)]


def get_normalised_iqr(y):
    y = np.array(y)
    iqr = np.percentile(y, 75) - np.percentile(y, 25)
    median = abs(np.median(y))
    return iqr / (median + epsilon)

def get_normalised_iqr_from_quantiles(quantile_vals, quantile_list):
    q1 = quantile_vals[quantile_list.index(0.25)]
    q3 = quantile_vals[quantile_list.index(0.75)]
    median = abs(quantile_vals[quantile_list.index(0.5)])
    iqr = q3 - q1
    return iqr / (median + epsilon)


def get_coverage_intervals(y, pred_quantiles, quantile_list, ci=95):
    y = np.array(y)
    lower_idx = quantile_list.index(ci_to_quantiles[str(ci)][0])
    upper_idx = quantile_list.index(ci_to_quantiles[str(ci)][1])
    lower_bound = pred_quantiles[lower_idx]
    upper_bound = pred_quantiles[upper_idx]
    return ((y > lower_bound) & (y < upper_bound)).mean()