"""Unit test split.py"""
import os
import sys
import numpy as np
import unittest
import pandas as pd
import random
from portfolioML.model.split import split_sequences, get_train_set, split_Tperiod

def _full_path(file_name):
    return os.path.join(os.path.abspath(os.path.dirname(__file__)), file_name)

df_return = pd.read_csv(_full_path('ReturnsData.csv'))
df_binary = pd.read_csv(_full_path('ReturnsBinary.csv'))

class TestDataReturns(unittest.TestCase):

    def test_Tperiod(self):
        """Test of Tperiod fuction, each study period must have the same lenght"""
        per_ret, per_bin = split_Tperiod(df_return, df_binary)
        check_value = len(per_ret[0])
        for elm1, elm2 in zip(per_ret, per_bin):
            self.assertAlmostEqual(check_value, len(elm1))
            self.assertAlmostEqual(check_value, len(elm2))



    def test_split_sequences(self):
        """Test of split_sequences fuction"""
        self.assertEqual(len(df_return), len(df_binary))
        column_idx = random.randint(1, df_return.shape[1])
        random_tick = df_binary.columns[column_idx]
        X, y = split_sequences(df_return[random_tick], df_binary[random_tick])
        for i in range(1,10):
            self.assertAlmostEqual(X[i][0], X[i-1][1])

    def test_get_train_set(self):
        """
        Test of get_train_set fuction.
        The check is maded compare the result of np.syack and the results 
        of get_trai_set when we use the reshape of numpy.
        """

        #nstack
        list1 = []
        list1
        for col in df_return.columns[1:10]:
            x1, y1 = split_sequences(df_return[col], df_binary[col])
            list1.append(x1)
        list1 = np.array(list1)
        list1 = list((list1[i] for i in range(list1.shape[0])))
        list1 = np.vstack(list1)

        #reshape
        list2, y2 = get_train_set(df_return, df_binary)

        self.assertTrue((list1 == list2[:9*6300]).all())



if __name__ == '__main__':
    unittest.main()