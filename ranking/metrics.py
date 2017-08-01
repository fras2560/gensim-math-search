'''
Created on Aug 1, 2017

@author: d6fraser
'''
from scipy.stats import entropy
from numpy.linalg import norm
from gensim.matutils import sparse2full
from testing.test_pipeline import TestIndexer
import numpy as np
import unittest


def jensen_shannon_divergence(vec1, vec2, model):
    """Written by Doug Shore

    Source:
    https://stackoverflow.com/questions/15880133/jensen-shannon-divergence
    """
    P = sparse2full(vec1, model.num_topics)
    Q = sparse2full(vec2, model.num_topics)
    _P = P / norm(P, ord=1)
    _Q = Q / norm(Q, ord=1)
    _M = 0.5 * (_P + _Q)
    return 0.5 * (entropy(_P, _M) + entropy(_Q, _M))


def hellinger_distance(vec1, vec2, model):
    """Written by Radim

    Source:
    https://stackoverflow.com/questions/22433884/python-gensim-how-to-calculate-document-similarity-using-the-lda-model
    """
    dense1 = sparse2full(vec1, model.num_topics)
    dense2 = sparse2full(vec2, model.num_topics)
    sim = np.sqrt(0.5 * ((np.sqrt(dense1) - np.sqrt(dense2))**2).sum())
    return sim


class Test(unittest.TestCase):
    def setUp(self):
        self.i1 = [[1, 0], [0, 1]]
        self.ni1 = [[0, 1], [1, 0]]
        self.i2 = [[2, 0], [0, 2]]
        self.all1 = [[1, 1], [1, 1]]

    def tearDown(self):
        pass

    def testJensenShannonDivergence(self):
        

if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
