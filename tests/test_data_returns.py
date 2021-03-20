"""Unit test data_returns"""
import unittest
import sys 
import pandas as pd
sys.path.append('../PortfolioML/Data')
import data_returns

dataframe = pd.read_csv('../PortfolioML/Data/PriceData.csv')
class TestDataRaturns(unittest.TestCase):
    """Class test for data_returns.py"""

    def test_m(self):
        """test correst result m"""
        m_list = range(1,10)
        for m in m_list:
            self.assertAlmostEqual(data_returns.get_returns(dataframe,m).ALL[0],(dataframe.ALL[m]/dataframe.ALL[0])-1)

if __name__ == '__main__':
    unittest.main()