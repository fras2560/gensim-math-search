'''
Created on Jul 24, 2017

@author: d6fraser
'''
from gensim import similarities, models, corpora
from create_models import create_model
from math_corpus import format_paragraph
from nltk.stem.porter import PorterStemmer
import unittest
import os
import shutil
import argparse
import logging
logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)
logging.root.level = logging.INFO


def create_index(corpus_path,
                 output_path,
                 model_path,
                 name,
                 lda=False,
                 lsi=False,
                 tfidf=False,
                 hdp=False):
    """Creates an index specified by the parameters & saves to output directory

    Parameters:
        corpus_path: the path to the corpus directory (os.path)
        output_path: the directory path where index(s) will be saved (os.path)
                     Note indexes each need their own folder
        model_path: the directory path with the models to be used (os.path)
                    The model path should have a corpus.dict and corpus.mm too
                    Use create_models.py
        name: the name of the index (str)
        lda: if True will create an index based on the lda model (boolean)
        lsi: if True will create an index based on the lsi model (boolean)
        tfidf: if True will create an index based on the tfidf model (boolean)
        hdp: if True will create an index based on hdp model (boolean)
    """
    dictionary = corpora.Dictionary.load(os.path.join(model_path,
                                                      "corpus.dict"))
    mc = corpora.MmCorpus(os.path.join(model_path, "corpus.mm"))
    # depending on the model the number of features changes
    if tfidf:
        model = models.TfidfModel.load(os.path.join(model_path, "model.tfidf"))
        op = os.path.join(output_path, name + "tfidf")
        index = similarities.Similarity(op,
                                        model[mc],
                                        num_features=len(dictionary))
        index.save(os.path.join(output_path, name + "-tfidf.index"))
    if lda:
        model = models.LdaModel.load(os.path.join(model_path, "model.lda"))
        op = os.path.join(output_path, name + "lda")
        index = similarities.Similarity(op,
                                        model[mc],
                                        num_features=model.num_topics)
        index.save(os.path.join(output_path, name + "-lda.index"))
    if lsi:
        model = models.LsiModel.load(os.path.join(model_path, "model.lsi"))
        op = os.path.join(output_path, name + "lsi")
        index = similarities.Similarity(op,
                                        model[mc],
                                        num_features=len(dictionary))
        index.save(os.path.join(output_path, name + "-lsi.index"))
    if hdp:
        model = models.HdpModel.load(os.path.join(model_path, "model.hdp"))
        op = os.path.join(output_path, name + "hdp")
        index = similarities.Similarity(op,
                                        model[mc],
                                        num_features=len(dictionary))
        index.save(os.path.join(output_path, name + "-hdp.index"))


class ModelException(Exception):
    pass


class Test(unittest.TestCase):
    def setUp(self):
        self.debug = True
        self.corpus = os.path.join(os.getcwd(), "tutorialDocuments")
        self.output = os.path.join(os.getcwd(), "testIndex")
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
        expected = [(2, 0.99994278),
                    (0, 0.99994081),
                    (3, 0.999879),
                    (4, 0.99935204),
                    (1, 0.99467087),
                    (8, 0.1938726),
                    (7, -0.023664713),
                    (6, -0.0515742),
                    (5, -0.088042185)]
        self.log(sims)
        for index, t in enumerate(sims):
            self.assertEqual(expected[index][0], t[0])
            self.assertAlmostEqual(expected[index][1], t[1])

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
    descp = """
            Create Gensim index that can be used to search
            for Mathematical Documents by using a model (see create_models)
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
    prompt = "The path to Model directory (created by create_model)"
    parser.add_argument("model",
                        help=prompt,
                        action="store")
    parser.add_argument("output",
                        help="The path to directory where model will be saved",
                        action="store")
    parser.add_argument("name",
                        help="The name of the index")
    args = parser.parse_args()
    # need to load the model
    create_index(args.corpus,
                 args.output,
                 args.model,
                 args.name,
                 lda=args.lda,
                 lsi=args.lsi,
                 tfidf=args.tfidf,
                 hdp=args.hdp
                 )
