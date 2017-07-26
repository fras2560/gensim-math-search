'''
Created on Jul 24, 2017

@author: d6fraser
'''
from math_corpus import MathCorpus
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
logging.root.level = logging.INFO  # ipython sometimes messes up the logging setup; restore


def create_index(corpus_path,
                 output_path,
                 model_path,
                 name,
                 lda=False,
                 lsi=False,
                 tfidf=False,
                 hdp=False,
                 dictionary=None):
    """Creates an index specified by the parameters & saves to output directory

    Parameters:
        corpus_path: the path to the corpus directory (os.path)
        output_path: the directory path where index(s) will be saved (os.path)
        model_path: the directory path with the models to be used (os.path)
        name: the name of the index (str)
    """
    dictionary = corpora.Dictionary.load(os.path.join(model_path,
                                                      "corpus.dict"))
    mc = corpora.MmCorpus(os.path.join(model_path, "corpus.mm"))
    # depending on the model the number of features changes
    if tfidf:
        model = models.TfidfModel.load(os.path.join(model_path, "model.tfidf"))
        index = similarities.Similarity(output_path,
                                        model[mc],
                                        num_features=len(dictionary))
        index.save(os.path.join(output_path, name + "-tfidf.index"))
    if lda:
        model = models.LdaModel.load(os.path.join(model_path, "model.lda"))
        index = similarities.Similarity(output_path,
                                        model[mc],
                                        num_features=model.num_topics)
        index.save(os.path.join(output_path, name + "-lda.index"))
    if lsi:
        model = models.LsiModel.load(os.path.join(model_path, "model.lsi"))
        index = similarities.Similarity(output_path,
                                        model[mc],
                                        num_features=len(dictionary))
        index.save(os.path.join(output_path, name + "-lsi.index"))
    if hdp:
        model = models.HdpModel.load(os.path.join(model_path, "model.hdp"))
        index = similarities.Similarity(output_path,
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

    def testLDA(self):
        create_index(self.corpus, self.output, self.output, "test", lda=True)
        index = similarities.Similarity.load(os.path.join(self.output,
                                                          "test-lda.index"))
        p = "(stored under {})".format(str(self.output))
        expect = "Similarity index with 9 documents in 1 shards {}".format(p)
        self.assertEqual(expect, str(index))

    def testLSI(self):
        create_index(self.corpus, self.output, self.output, "test", lsi=True)
        index = similarities.Similarity.load(os.path.join(self.output,
                                                          "test-lsi.index"))
        p = "(stored under {})".format(str(self.output))
        expect = "Similarity index with 9 documents in 1 shards {}".format(p)
        self.assertEqual(expect, str(index))
        # search with the index
        doc = "Human computer interaction"
        lsi = models.LdaModel.load(os.path.join(self.output, "model.lsi"))
        print(format_paragraph(doc, PorterStemmer()))
        vec_bow = self.dictionary.doc2bow(format_paragraph(doc,
                                                           PorterStemmer()))
        self.log(lsi)
        print(lsi, vec_bow)
        vec_lsi = lsi[vec_bow]
        sims = index[vec_lsi]
        sims = sorted(enumerate(sims), key=lambda item: -item[1])
        print(sims)

    def testHDP(self):
        create_index(self.corpus, self.output, self.output, "test", hdp=True)
        index = similarities.Similarity.load(os.path.join(self.output,
                                                          "test-hdp.index"))
        p = "(stored under {})".format(str(self.output))
        expect = "Similarity index with 9 documents in 1 shards {}".format(p)
        self.assertEqual(expect, str(index))

    def testTFIDF(self):
        create_index(self.corpus, self.output, self.output, "test", tfidf=True)
        index = similarities.Similarity.load(os.path.join(self.output,
                                                          "test-tfidf.index"))
        p = "(stored under {})".format(str(self.output))
        expect = "Similarity index with 9 documents in 1 shards {}".format(p)
        self.assertEqual(expect, str(index))


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
    prompt = "The path to Model directory (created by create_index)"
    parser.add_argument("model",
                        help=prompt,
                        action="store")
    parser.add_argument("output",
                        help="The path to directory where model will be saved",
                        action="store")
    parser.add_argument("name",
                        help="The name of the index")
    parser.add_argument("-d", "--dictionary",
                        help="The filepath to the saved dictionary",
                        dest="dictionary",
                        default=None)
    args = parser.parse_args()
    # need to load the model
    create_index(args.corpus,
                 args.output,
                 args.model,
                 args.name,
                 dictionary=args.dictionary)
