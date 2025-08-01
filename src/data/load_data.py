import pandas as pd

def load_lol_data(filepath='League_of_Legends_Ranked_Match_Data.csv'):
    return pd.read_csv(filepath)
