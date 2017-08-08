'''
Created on Aug 1, 2017

@author: d6fraser
'''
from scipy.stats import entropy
from numpy.linalg import norm
from gensim.matutils import sparse2full
import numpy as np


def jensen_shannon_divergence(vec1, vec2, model):
    """Written by Doug Shore

    Source:
    https://stackoverflow.com/questions/15880133/jensen-shannon-divergence
    """
    try:
        num_topics = model.num_topics
    except AttributeError:
        num_topics = len(model.id2word)
    P = sparse2full(vec1, num_topics)
    Q = sparse2full(vec2, num_topics)
    _P = P / norm(P, ord=1)
    _Q = Q / norm(Q, ord=1)
    _M = 0.5 * (_P + _Q)
    return 0.5 * (entropy(_P, _M) + entropy(_Q, _M))


def hellinger_distance(vec1, vec2, model):
    """Written by Radim

    Source:
    https://stackoverflow.com/questions/22433884/python-gensim-how-to-calculate-document-similarity-using-the-lda-model
    """
    try:
        num_topics = model.num_topics
    except AttributeError:
        num_topics = len(model.id2word)
    dense1 = sparse2full(vec1, num_topics)
    dense2 = sparse2full(vec2, num_topics)
    sim = np.sqrt(0.5 * ((np.sqrt(dense1) - np.sqrt(dense2))**2).sum())
    return sim
