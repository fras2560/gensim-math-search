'''
Created on Jul 20, 2017

@author: d6fraser
'''

from math_corpus import MathCorpus, format_paragraph
from gensim import models, corpora
from nltk.stem.porter import PorterStemmer
import os
import unittest
import shutil
import argparse
import logging
logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)
logging.root.level = logging.INFO


def create_model(corpus_path,
                 output_path,
                 num_topics=500,
                 tfidf=False,
                 lda=False,
                 lsi=False,
                 hdp=False):
    """Creates a model(s) specify by the parameters and save to output directory

    Parameters:
        corpus_path: the path to the corpus directory (os.path)
        output_path: the directory path where model(s) will be saved (os.path)
        tfidf=False: True if want a tfidf model created (boolean)
        lda=False: True if want a lda model created (boolean)
        lsi=False: True if want a lsi model created (boolean)
        hdp=False: True if want a hdp model created (boolean)
    """
    mc = MathCorpus(corpus_path)
    mc.save_dictionary(os.path.join(output_path, "corpus.dict"))
    corpora.MmCorpus.serialize(os.path.join(output_path, "corpus.mm"), mc)
    tfidf_model = models.TfidfModel(mc)
    if tfidf:
        tfidf_model.save(os.path.join(output_path, "model.tfidf"))
    if lda:
        lda_model = models.LdaModel(mc,
                                    id2word=mc.dictionary,
                                    num_topics=num_topics)
        lda_model.save(os.path.join(output_path, "model.lda"))
    if lsi:
        lsi_model = models.LsiModel(tfidf_model[mc],
                                    id2word=mc.dictionary,
                                    num_topics=num_topics)
        lsi_model.save(os.path.join(output_path, "model.lsi"))
    if hdp:
        hdi_model = models.HdpModel(mc, id2word=mc.dictionary)
        hdi_model.save(os.path.join(output_path, "model.hdp"))


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

    def reverse_lookup(self, dictionary, value):
        for key, v in dictionary.items():
            if value == v:
                return key

    def format_lsi(self, expect):
        return " + ".join(["{:.3f}".format(t[0]) +
                           '*"{}"'.format(t[1])
                           for t in expect])

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
        create_model(self.corpus,
                     self.output,
                     num_topics=2,
                     lda=True)
        # check the model
        path = os.path.join(self.output, "model.lda")
        lda_model = models.LdaModel.load(path)
        self.log(lda_model)
        doc_bow = [(0, 1), (1, 1)]
        answer = lda_model[doc_bow]
        self.log(answer)
        self.assertAlmostEqual(len(answer), 4, delta=3)

    def testLSI(self):
        create_model(self.corpus, self.output, num_topics=2, lsi=True)
        # check the model
        path = os.path.join(self.output, "model.lsi")
        lsi_model = models.LsiModel.load(path)
        self.log(lsi_model)
        doc_bow = [(0, 1), (1, 1)]
        answer = lsi_model[doc_bow]
        self.log(answer)
        self.assertAlmostEqual(len(answer), 2, delta=1)
        # make sure can vev bow the document
        doc = "Human computer interaction"
        self.dictionary = corpora.Dictionary.load(os.path.join(self.output,
                                                               "corpus.dict"))
        vec_bow = self.dictionary.doc2bow(format_paragraph(doc,
                                                           PorterStemmer()))
        self.log(lsi_model)
        corpus = corpora.MmCorpus(os.path.join(self.output, "corpus.mm"))
        vec_lsi = lsi_model[vec_bow]
        self.assertEqual(len(vec_lsi), 2)
        e1_1 = [(0.703, "tree"),
                (0.538, "graph"),
                (0.402, "minor"),
                (0.187, "survey"),
                (0.061, "system"),
                (0.060, "time"),
                (0.060, "respons"),
                (0.058, "user"),
                (0.049, "comput"),
                (0.035, "interfac")]
        e1_2 = [(0.703, "tree"),
                (0.538, "graph"),
                (0.402, "minor"),
                (0.187, "survey"),
                (0.061, "system"),
                (0.060, "respons"),
                (0.060, "time"),
                (0.058, "user"),
                (0.049, "comput"),
                (0.035, "interfac")]
        e1_3 = [(-t[0], t[1]) for t in e1_1]
        e1_4 = [(-t[0], t[1]) for t in e1_2]
        e2_1 = [(0.460, "system"),
                (0.373, "user"),
                (0.332, "ep"),
                (0.328, "interfac"),
                (0.320, "respons"),
                (0.320, "time"),
                (0.293, "comput"),
                (0.280, "human"),
                (0.171, "survey"),
                (-0.161, "tree")]
        e2_2 = [(0.460, "system"),
                (0.373, "user"),
                (0.332, "ep"),
                (0.328, "interfac"),
                (0.320, "time"),
                (0.320, "respons"),
                (0.293, "comput"),
                (0.280, "human"),
                (0.171, "survey"),
                (-0.161, "tree")]
        e2_3 = [(-t[0], t[1]) for t in e2_1]
        e2_4 = [(-t[0], t[1]) for t in e2_2]
        expect = [[self.format_lsi(e1_1),
                   self.format_lsi(e1_2),
                   self.format_lsi(e1_3),
                   self.format_lsi(e1_4)],
                  [self.format_lsi(e2_1),
                   self.format_lsi(e2_2),
                   self.format_lsi(e2_3),
                   self.format_lsi(e2_4)]
                  ]
        for index, values in enumerate(lsi_model.print_topics()):
            self.assertEqual(values[1] in expect[index], True)

    def testHDP(self):
        create_model(self.corpus, self.output, num_topics=2, hdp=True)
        # check the model
        path = os.path.join(self.output, "model.hdp")
        hdp_model = models.LsiModel.load(path)
        self.log(hdp_model)
        doc_bow = [(0, 1), (1, 1)]
        answer = hdp_model[doc_bow]
        self.log(answer)
        self.assertAlmostEqual(len(answer), 6, delta=3)

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
    parser.add_argument('-hdp',
                        dest="hdp",
                        action="store_true",
                        help="Build HDP Model",
                        default=False)
    parser.add_argument("corpus",
                        help="The path to Math Corpus directory (html, xhtml)",
                        action="store")
    parser.add_argument("output",
                        help="The path to directory where model will be saved",
                        action="store")
    args = parser.parse_args()
    create_model(args.corpus,
                 args.output,
                 tfidf=args.tfidf,
                 lda=args.lda,
                 lsi=args.lsi,
                 hdp=args.hdp)
