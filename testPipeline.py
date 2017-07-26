'''
Created on Jul 20, 2017

@author: d6fraser
'''
from math_corpus import MathCorpus
from create_models import create_model, TestCreateModels
from query import Indexer, DocumentCollection, ArxivQueries,\
                  ExpectedResults, Query
from bs4 import BeautifulSoup
from gensim import models, similarities, corpora
import unittest
import os
import shutil
from create_index import create_index


class TestQuery(unittest.TestCase):
    def setUp(self):
        self.topic = None
        with open(os.path.join(os.getcwd(), "test", "testQuery.xml")) as doc:
            soup = BeautifulSoup(doc)
            topic = soup.find("topic")
            self.topic = topic

    def tearDown(self):
        pass

    def testConstructor(self):
        q = Query(self.topic)
        self.assertEqual("NTCIR12-MathWiki-1", str(q))
        expect = " symbol  ('v!ζ','!0','n')"
        self.assertEqual(expect, q.get_words())


class TestExpectedResults(unittest.TestCase):
    def setUp(self):
        self.results = os.path.join(os.getcwd(), "test", "simple-results.dat")

    def tearDown(self):
        pass

    def testExpectedResultsConstructor(self):
        with open(self.results) as document:
            e_r = ExpectedResults(document)
            expect = {"NTCIR12-MathIR-10": {'0705.0010_1_359': 2.0,
                                            '0705.2373_1_20': 0.0},
                      "NTCIR12-MathIR-1": {'quant-ph9904101_1_41': 3.0,
                                           '0710.3032_1_22': 1.0},
                      "NTCIR12-MathIR-2": {'0705.4299_1_46': 1.0,
                                           '0705.4299_1_56': 0.0}
                      }
            self.assertEqual(expect, e_r.results)

    def testFindScore(self):
        with open(self.results) as document:
            e_r = ExpectedResults(document)
            # missing query
            score = e_r.find_score("NoQuery", "0705.0010_1_359")
            self.assertEqual(score, -1)
            # missing document
            score = e_r.find_score("NTCIR12-MathIR-10", "NoDocument")
            self.assertEqual(score, -1)
            # a hit
            score = e_r.find_score("NTCIR12-MathIR-10", "0705.0010_1_359")
            print(e_r.results)
            self.assertEqual(score, 2.0)

    def testParseName(self):
        with open(self.results) as document:
            e_r = ExpectedResults(document)
            name = e_r.parse_name("0705.0010_1_359.xhtml")
            self.assertEqual(name, "0705.0010_1_359")
            name = e_r.parse_name("0705.0010_1_359")
            self.assertEqual(name, "0705.0010_1_359")


class TestDocumentCollection(unittest.TestCase):
    def setUp(self):
        self.directory = os.path.join(os.getcwd(), "tutorialDocuments")
        self.dc = DocumentCollection(self.directory)

    def tearDown(self):
        pass

    def testLookup(self):
        result = self.dc.lookup(1)
        expect = os.path.join(self.directory, "2.html")
        self.assertEqual(result, str(expect))

    def testConstructor(self):
        expect = {
                 0: str(os.path.join(self.directory, '1.html')),
                 1: str(os.path.join(self.directory, '2.html')),
                 2: str(os.path.join(self.directory, '3.html')),
                 3: str(os.path.join(self.directory, '4.html')),
                 4: str(os.path.join(self.directory, '5.html')),
                 5: str(os.path.join(self.directory, '6.html')),
                 6: str(os.path.join(self.directory, '7.html')),
                 7: str(os.path.join(self.directory, '8.html')),
                 8: str(os.path.join(self.directory, '9.html'))
                 }
        self.assertEqual(self.dc.mapping, expect)


class TestIndexer(unittest.TestCase):
    def setUp(self):
        self.debug = True
        self.corpus = os.path.join(os.getcwd(), "tutorialDocuments")
        self.models = os.path.join(os.getcwd(), "testModels")
        self.index = os.path.join(os.getcwd(), "testIndex")
        if not os.path.exists(self.models):
            os.makedirs(self.models)
        else:
            shutil.rmtree(self.models)
            os.makedirs(self.models)
        if not os.path.exists(self.index):
            os.makedirs(self.index)
        else:
            shutil.rmtree(self.index)
            os.makedirs(self.index)
        create_model(self.corpus,
                     self.models,
                     num_topics=2,
                     lda=True,
                     lsi=True,
                     tfidf=True,
                     hdp=True)
        # load up the models
        self.dictionary = corpora.Dictionary.load(os.path.join(self.models,
                                                               "corpus.dict"))
        self.corp = corpora.MmCorpus(os.path.join(self.models,
                                                  "corpus.mm"))
        path = os.path.join(self.models, "model.lda")
        self.lda = models.LdaModel.load(path)
        path = os.path.join(self.models, "model.lsi")
        self.lsi = models.LsiModel.load(path)
        path = os.path.join(self.models, "model.tfidf")
        self.tfidf = models.TfidfModel.load(path)
        path = os.path.join(self.models, "model.hdp")
        self.hdp = models.HdpModel.load(path)
        # create the indexes
        create_index(self.corpus,
                     self.index,
                     self.models,
                     "test",
                     lda=True,
                     lsi=True,
                     tfidf=True,
                     hdp=True)
        # load the indexes
        path = os.path.join(self.index, "test-lsi.index")
        self.lsi_index = similarities.Similarity.load(path)
        path = os.path.join(self.index, "test-lda.index")
        self.lda_index = similarities.Similarity.load(path)
        path = os.path.join(self.index, "test-tfidf.index")
        self.tfidf_index = similarities.Similarity.load(path)
        path = os.path.join(self.index, "test-hdp.index")
        self.hdp_index = similarities.Similarity.load(path)

    def log(self, message):
        if self.debug:
            print(message)

    def tearDown(self):
        if os.path.exists(self.models):
            shutil.rmtree(self.models)
        if os.path.exists(self.index):
            shutil.rmtree(self.index)

    def testSearchLSI(self):
        indexer = Indexer(self.dictionary,
                          self.lsi,
                          self.lsi_index,
                          self.corpus)
        doc = """
                <num>test-query</num>
                <keyword>Human computer interaction<keyword>
              """
        query = Query(BeautifulSoup(doc))
        results = indexer.search(query)
        expect = [os.path.join(self.corpus, '1.html'),
                  os.path.join(self.corpus, '4.html'),
                  os.path.join(self.corpus, '2.html'),
                  os.path.join(self.corpus, '6.html'),
                  os.path.join(self.corpus, '7.html'),
                  os.path.join(self.corpus, '8.html'),
                  os.path.join(self.corpus, '9.html'),
                  os.path.join(self.corpus, '3.html'),
                  os.path.join(self.corpus, '5.html')
                  ]
        self.debug = True
        self.log(expect)
        self.log(results)
        print("Topics")
        self.lsi.print_debug(2)
        print(self.lsi)
        for i in self.lsi.print_topics():
            print(i)
        self.assertEqual(expect, results)
        doc = """
                <num>test-query</num>
                <keyword>Tree orderings<keyword>
              """
        query = Query(BeautifulSoup(doc))
        results = indexer.search(query)
        expect = [os.path.join(self.corpus, '6.html'),
                  os.path.join(self.corpus, '7.html'),
                  os.path.join(self.corpus, '8.html'),
                  os.path.join(self.corpus, '4.html'),
                  os.path.join(self.corpus, '1.html'),
                  os.path.join(self.corpus, '5.html'),
                  os.path.join(self.corpus, '2.html'),
                  os.path.join(self.corpus, '3.html'),
                  os.path.join(self.corpus, '9.html')
                  ]
        self.assertEqual(expect, results)

    def testSearchTFIDF(self):
        indexer = Indexer(MathCorpus(self.corpus).dictionary,
                          self.tfidf,
                          self.tfidf_index,
                          self.corpus)
        doc = """
                <num>test-query</num>
                <keyword>Human computer interaction<keyword>
              """
        query = Query(BeautifulSoup(doc))
        results = indexer.search(query)
        expect = [os.path.join(self.corpus, '1.html'),
                  os.path.join(self.corpus, '4.html'),
                  os.path.join(self.corpus, '2.html')]
        self.log(results)
        self.assertEqual(results, expect)
        doc = """
                <num>test-query</num>
                <keyword>tree ordering<keyword>
              """
        query = Query(BeautifulSoup(doc))
        results = indexer.search(query)
        expect = [os.path.join(self.corpus, '6.html'),
                  os.path.join(self.corpus, '7.html'),
                  os.path.join(self.corpus, '8.html')]
        self.assertEqual(results, expect)

if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
