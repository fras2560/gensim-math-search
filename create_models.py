'''
Created on Jul 20, 2017

@author: d6fraser
'''
from math_corpus import MathCorpus
from gensim import models
import os
import unittest
import shutil
import argparse


def create_model(corpus_path, output_path, tfidf=False, lda=False, lsi=False):
    """Creates a model(s) specify by the parameters and save to output directory

    Parameters:
        corpus_path: the path to the corpus directory (os.path)
        output_path: the directory path where model(s) will be saved (os.path)
        tfidf=False: True if want a tfidf model created (boolean)
        lda=False: True if want a lda model created (boolean)
        lsi=False: True if want a lsi model created (boolean)
    """
    mc = MathCorpus(corpus_path)
    tfidf_model = models.TfidfModel(mc)
    if tfidf:
        tfidf_model.save(os.path.join(output_path, "model.tfidf"))
    if lda:
        lda_model = models.LdaModel(mc, id2word=mc.dictionary, num_topics=500)
        lda_model.save(os.path.join(output_path, "model.lda"))
    if lsi:
        lsi_model = models.LsiModel(tfidf_model[mc],
                                    id2word=mc.dictionary,
                                    num_topics=500)
        lsi_model.save(os.path.join(output_path, "model.lsi"))


class TestCreateModels(unittest.TestCase):
    def setUp(self):
        self.debug = True
        self.corpus = os.path.join(os.getcwd(), "tutorialDocuments")
        self.output = os.path.join(os.getcwd(), "testModels")
        if not os.path.exists(self.output):
            os.makedirs(self.output)
        else:
            shutil.rmtree(self.output)
            os.makedirs(self.output)

    def log(self, message):
        if self.debug:
            print(message)

    def tearDown(self):
        if os.path.exists(self.output):
            shutil.rmtree(self.output)

    def testTFIDF(self):
        create_model(self.corpus, self.output, tfidf=True)
        # check the model
        path = os.path.join(self.output, "model.tfidf")
        tfidf_model = models.TfidfModel.load(path)
        self.log(tfidf_model)
        doc_bow = [(0, 1), (1, 1)]
        answer = tfidf_model[doc_bow]
        self.log(answer)
        self.assertAlmostEqual(0, answer[0][0])
        self.assertAlmostEqual(1, answer[1][0])

    def testLDA(self):
        create_model(self.corpus, self.output, lda=True)
        # check the model
        path = os.path.join(self.output, "model.lda")
        lda_model = models.LdaModel.load(path)
        self.log(lda_model)
        doc_bow = [(0, 1), (1, 1)]
        answer = lda_model[doc_bow]
        self.log(answer)
        self.assertAlmostEqual(len(answer), 4, delta=3)

    def testLSI(self):
        create_model(self.corpus, self.output, lsi=True)
        # check the model
        path = os.path.join(self.output, "model.lsi")
        lsi_model = models.LsiModel.load(path)
        self.log(lsi_model)
        doc_bow = [(0, 1), (1, 1)]
        answer = lsi_model[doc_bow]
        self.log(answer)
        self.assertAlmostEqual(len(answer), 6, delta=2)

if __name__ == "__main__":
    descp = """
            Create Mathematical Topic Models using Gensim
            Author: Dallas Fraser (d6fraser@uwaterloo.ca)
            """
    parser = argparse.ArgumentParser(description=descp)
    parser.add_argument('-lsi',
                        dest="lsi",
                        action="store_true",
                        default=False,
                        help="Build LSI Model")
    parser.add_argument('-lda',
                        dest="lda",
                        action="store_true",
                        help="Build LDA Model",
                        default=False)
    parser.add_argument('-tfidf',
                        dest="tfidf",
                        action="store_true",
                        help="Build TFIDF Model",
                        default=False)
    parser.add_argument("corpus",
                        help="The path to Math Corpus directory (html, xhtml)",
                        action="store")
    parser.add_argument("output",
                        help="The path to directory where model will be saved",
                        action="store")
    args = parser.parse_args()
    tfidf = False
    lda = False
    lsi = False
    if args.tfidf:
        tfidf = True
    if args.lda:
        lda = True
    if args.lsi:
        lsi = True
    create_model(args.corpus, args.output, tfidf=tfidf, lda=lda, lsi=lsi)
