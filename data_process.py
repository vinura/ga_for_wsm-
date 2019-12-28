import numpy as np
import pandas as pd

from sklearn import preprocessing

def preprocess():
    data = pd.read_csv('Data.csv')

    x = data[['NUMPOINTS','Elev_mean','Popsum','CoverageMe']].values
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df = pd.DataFrame(x_scaled)
    df[['id','Restricted']] = data[['id','Restricted']]
    df.rename(columns={0:'NUMPOINTS',1:'Elev_mean', 2:'Popsum', 3:'CoverageMe'}, inplace=True)
    df.set_index('id', inplace=True)

    return df

