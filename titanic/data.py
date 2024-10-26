import numpy as np
import pandas as pd

def missing_data_table_analysis(data):
    """
    Create a table containing total number and percent of missing values for each column.
    """
    total = data.isnull().sum()
    percent = data.isnull().sum() / data.isnull().count() * 100
    tt = pd.concat([total, percent], axis=1, keys=["Total", "Percent"])
    types = []
    for col in data.columns:
        dtype = str(data[col].dtype)
        types.append(dtype)
    tt["Types"] = types
    df_missing = np.transpose(tt)

    return df_missing
