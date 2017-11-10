'''
Name: Dallas Fraser
Email: d6fraser@uwaterloo.ca
Date: 2017-07-27
Project: Tangent GT
Purpose: Tests create_models function
'''
from scripts.create_models import create_model
from tangent.math_corpus import format_paragraph
from gensim import corpora, models
from nltk.stem.porter import PorterStemmer
import unittest
import os
import shutil


class TestCreateModels(unittest.TestCase):
    def setUp(self):
        self.debug = True
        cwd = os.path.dirname(os.getcwd())
        self.corpus = os.path.join(cwd, "testing", "test", "tutorialDocuments")
        self.output = os.path.join(cwd, "testModels")
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
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
