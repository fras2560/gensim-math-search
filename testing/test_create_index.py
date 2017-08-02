'''
Name: Dallas Fraser
Email: d6fraser@uwaterloo.ca
Date: 2017-07-27
Project: Tangent GT
Purpose: Tests the create_index function
'''
from gensim import similarities, models, corpora
from scripts.create_models import create_model
from scripts.create_index import create_index
from tangent.math_corpus import format_paragraph
from nltk.stem.porter import PorterStemmer
import unittest
import os
import shutil


class Test(unittest.TestCase):
    def setUp(self):
        self.debug = True
        cwd = os.path.dirname(os.getcwd())
        self.corpus = os.path.join(cwd, "testing", "test", "tutorialDocuments")
        self.output = os.path.join(cwd, "testIndex")
        if not os.path.exists(self.output):
            os.makedirs(self.output)
        else:
            shutil.rmtree(self.output)
            os.makedirs(self.output)
        create_model(self.corpus,
                     self.output,
                     num_topics=2,
                     lda=True,
                     lsi=True,
                     tfidf=True,
                     hdp=True)
        self.dictionary = corpora.Dictionary.load(os.path.join(self.output,
                                                               "corpus.dict"))

    def log(self, message):
        if self.debug:
            print(message)

    def tearDown(self):
        if os.path.exists(self.output):
            shutil.rmtree(self.output)
        if os.path.exists(os.path.join(os.getcwd(), "testIndex.0")):
            os.remove(os.path.join(os.getcwd(), "testIndex.0"))

    def testLDA(self):
        create_index(self.corpus, self.output, self.output, "test", lda=True)
        index = similarities.Similarity.load(os.path.join(self.output,
                                                          "test-lda.index"))
        op = os.path.join(self.output, "testlda")
        p = "(stored under {})".format(str(op))
        expect = "Similarity index with 9 documents in 1 shards {}".format(p)
        self.assertEqual(expect, str(index))

    def testLSI(self):
        lsi_model = models.LsiModel.load(os.path.join(self.output,
                                                      "model.lsi"))
        create_index(self.corpus, self.output, self.output, "test", lsi=True)
        index = similarities.Similarity.load(os.path.join(self.output,
                                                          "test-lsi.index"))
        op = os.path.join(self.output, "testlsi")
        p = "(stored under {})".format(str(op))
        expect = "Similarity index with 9 documents in 1 shards {}".format(p)
        self.assertEqual(expect, str(index))
        # search with the index
        doc = "Human computer interaction"
        vec_bow = self.dictionary.doc2bow(format_paragraph(doc,
                                                           PorterStemmer()))
        self.log(lsi_model)
        vec_lsi = lsi_model[vec_bow]
        sims = index[vec_lsi]
        sims = sorted(enumerate(sims), key=lambda item: -item[1])
        expected = [(0, 0.99994081),
                    (2, 0.99990785),
                    (3, 0.99984384),
                    (4, 0.9992786),
                    (1, 0.99330217),
                    (8, 0.22248439),
                    (7, -0.016480923),
                    (6, -0.0515742),
                    (5, -0.08804217)]
        self.log(sims)
        for index, t in enumerate(sims):
            self.assertEqual(expected[index][0], t[0])
            self.assertAlmostEqual(expected[index][1], t[1], delta=0.001)

    def testHDP(self):
        create_index(self.corpus, self.output, self.output, "test", hdp=True)
        index = similarities.Similarity.load(os.path.join(self.output,
                                                          "test-hdp.index"))
        op = os.path.join(self.output, "testhdp")
        p = "(stored under {})".format(str(op))
        expect = "Similarity index with 9 documents in 1 shards {}".format(p)
        self.assertEqual(expect, str(index))

    def testAllIndexes(self):
        tfidf_model = models.LsiModel.load(os.path.join(self.output,
                                                        "model.tfidf"))
        create_index(self.corpus,
                     self.output,
                     self.output,
                     "test",
                     tfidf=True,
                     lda=True,
                     lsi=True,
                     hdp=True)
        index = similarities.Similarity.load(os.path.join(self.output,
                                                          "test-tfidf.index"))
        op = os.path.join(self.output, "testtfidf")
        p = "(stored under {})".format(str(op))
        expect = "Similarity index with 9 documents in 1 shards {}".format(p)
        self.assertEqual(expect, str(index))
        doc = "Human computer interaction"
        vec_bow = self.dictionary.doc2bow(format_paragraph(doc,
                                                           PorterStemmer()))
        self.log(tfidf_model)
        vec_tfidf = tfidf_model[vec_bow]
        sims = index[vec_tfidf]
        print(sims)
        sims = sorted(enumerate(sims), key=lambda item: -item[1])
        expected = [(0, 0.81649655),
                    (3, 0.34777319),
                    (1, 0.31412902),
                    (2, 0.0),
                    (4, 0.0),
                    (5, 0.0),
                    (6, 0.0),
                    (7, 0.0),
                    (8, 0.0)]
        self.log(sims)
        for index, t in enumerate(sims):
            self.assertEqual(expected[index][0], t[0])
            self.assertAlmostEqual(expected[index][1], t[1])

    def testTFIDF(self):
        tfidf_model = models.LsiModel.load(os.path.join(self.output,
                                                        "model.tfidf"))
        create_index(self.corpus, self.output, self.output, "test", tfidf=True)
        index = similarities.Similarity.load(os.path.join(self.output,
                                                          "test-tfidf.index"))
        op = os.path.join(self.output, "testtfidf")
        p = "(stored under {})".format(str(op))
        expect = "Similarity index with 9 documents in 1 shards {}".format(p)
        self.assertEqual(expect, str(index))
        doc = "Human computer interaction"
        vec_bow = self.dictionary.doc2bow(format_paragraph(doc,
                                                           PorterStemmer()))
        self.log(tfidf_model)
        vec_tfidf = tfidf_model[vec_bow]
        sims = index[vec_tfidf]
        sims = sorted(enumerate(sims), key=lambda item: -item[1])
        expected = [(0, 0.81649655),
                    (3, 0.34777319),
                    (1, 0.31412902),
                    (2, 0.0),
                    (4, 0.0),
                    (5, 0.0),
                    (6, 0.0),
                    (7, 0.0),
                    (8, 0.0)]
        self.log(sims)
        for index, t in enumerate(sims):
            self.assertEqual(expected[index][0], t[0])
            self.assertAlmostEqual(expected[index][1], t[1])

if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
