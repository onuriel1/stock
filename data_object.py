import MySQLdb
import pandas as pd
import numpy as np


class DataObject():

    def __init__(self, table, mood_threshold=2):
        self.non_float_cols = []
        self.table = table
        self.data = None
        self.mood_threshold = mood_threshold

    def load_data(self):
        # Open database connection
        con = MySQLdb.connect("localhost", "stock", "qwe123qwe", "stock")
        query = "Select * from {0}".format(self.table)
        self.data = pd.read_sql(query, con)
        con.close()
        self.preprocess_data()

    def get_data(self, cols=None, not_cols=None, float_col_only=True):
        if cols is not None:
            return self.data.drop(self.non_float_cols, axis=1)[cols] if float_col_only else self.data.copy()[cols]
        if not_cols is not None:
            return self.data.drop(not_cols+self.non_float_cols, axis=1) if float_col_only else self.data.drop(not_cols, axis=1)
        return self.data.drop(self.non_float_cols, axis=1) if float_col_only else self.data.copy()

    def preprocess_data(self):
        self.data = self.data.apply(self.cast_to_float)
        # extract label
        self.data['label'] = ((self.data['CPriceA'] - self.data['OPriceB'])/self.data['OPriceB']*100).apply(lambda x: -1 if x < -self.mood_threshold else 1 if x > self.mood_threshold else 0)
        self.data['zeroVsrest'] = self.data['label'].apply(lambda x: 1 if x != 0 else 0)


    def cast_to_float(self, x):
        try:
            return np.float32(x)
        except:
            self.non_float_cols += [x.name]
            return x

    def get_labels(self):
        return self.data['label']

    def get_zero_vs_rest_labels(self):
        return self.data['zeroVsrest']

    def get_non_zeros(self, not_cols=None):
        if not_cols is None:
            not_cols = []
        return self.get_data_from_df(self.data[self.data['label'] != 0], not_cols=self.non_float_cols+not_cols), self.get_labels()[self.data['label'] !=0]

    @staticmethod
    def get_data_from_df(df, cols=None, not_cols=None):
        if cols:
            return df.drop(not_cols, axis=1)[cols]
        return df.drop(not_cols, axis=1)