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


def most_freq_table(data):
    """
    Create a frequency table.
    """
    total = data.count()
    tt = pd.DataFrame(total)
    tt.columns = ['Total']
    items = []
    vals = []
    for col in data.columns:
        try:
            itm = data[col].value_counts().index[0]
            val = data[col].value_counts().values[0]
            items.append(itm)
            vals.append(val)
        except Exception as ex:
            print(ex)
            items.append(0)
            vals.append(0)
            continue
    tt['Most frequent item'] = items
    tt['Frequence'] = vals
    tt['Percent from total'] = np.round(vals / total * 100, 3)
    np.transpose(tt)
    
    return tt


def unique_values_table(data):
    """
    Create a table with unique values.
    """
    total = data.count()
    tt = pd.DataFrame(total)
    tt.columns = ['Total']
    uniques = []
    for col in data.columns:
        unique = data[col].nunique()
        uniques.append(unique)
    tt['Uniques'] = uniques
    np.transpose(tt)

    return tt


def concatenator(df1,df2):
    """
    Concatenate two dataframes and add a column 'set' to identify test and train set.
    """
    all_df = pd.concat([df1, df2], axis=0)
    all_df["set"] = "train"
    all_df.loc[all_df.Survived.isna(), "set"] = "test"
    return all_df


def family_size(data):
    """
    Create a column containing the size of the family.
    """
    data["Family Size"] = data["SibSp"] + data["Parch"] + 1
    return data


def age_interval(data, age_col="Age"):
    """
    Split the age column into age intervals creating a new column called 'Age Interval'.
    """
    data["Age Interval"] = 0.0
    data.loc[data[age_col] <= 16, "Age Interval"] = 0
    data.loc[(data[age_col] > 16) & (data[age_col] <= 32), "Age Interval"] = 1
    data.loc[(data[age_col] > 32) & (data[age_col] <= 48), "Age Interval"] = 2
    data.loc[(data[age_col] > 48) & (data[age_col] <= 64), "Age Interval"] = 3
    data.loc[data[age_col] > 64, "Age Interval"] = 4
    return data


def fare_interval(data, fare_col="Fare"):
    """
    Split the Fare column into fare intervals creating a new column called 'Fare Interval'.
    """
    data["Fare Interval"] = 0.0
    data.loc[data[fare_col] <= 7.91, "Fare Interval"] = 0
    data.loc[(data[fare_col] > 7.91) & (data[fare_col] <= 14.454), "Fare Interval"] = 1
    data.loc[(data[fare_col] > 14.454) & (data[fare_col] <= 31), "Fare Interval"] = 2
    data.loc[data[fare_col] > 31, "Fare Interval"] = 3
    return data


def sex_pclass(data):
    """
    Create a column combining the column entries from Sex and Pclass column
    """
    data["Sex_Pclass"] = data.apply(
        lambda row: row["Sex"][0].upper() + "_C" + str(row["Pclass"]), axis=1
    )
    return data


def parse_names(row):
    """
    Extract name components (first name, family name, maiden name, title) from name.
    """
    try:
        text = row["Name"]
        split_text = text.split(",")
        family_name = split_text[0]
        next_text = split_text[1]
        split_text = next_text.split(".")
        title = (split_text[0] + ".").lstrip().rstrip()
        next_text = split_text[1]
        if "(" in next_text:
            split_text = next_text.split("(")
            given_name = split_text[0]
            maiden_name = split_text[1].rstrip(")")
            return pd.Series([family_name, title, given_name, maiden_name])
        else:
            given_name = next_text
            return pd.Series([family_name, title, given_name, None])
    except Exception as ex:
        print(f"Exception: {ex}")


def process_names(data):
    """
    Process name using the parser.
    """
    data[["Family Name", "Title", "Given Name", "Maiden Name"]] = data.apply(lambda row: parse_names(row), axis=1)
    return data