import pandas as pd

def load_lol_data(filepath='lol_data.csv'):
    return pd.read_csv(filepath)
