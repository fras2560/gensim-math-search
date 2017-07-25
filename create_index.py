'''
Created on Jul 24, 2017

@author: d6fraser
'''
from math_corpus import MathCorpus
from gensim import similarities, models
from create_models import create_model
import unittest
import os
import shutil
import argparse
import sys


def create_index(corpus_path, output_path, model, name):
    """Creates an index specified by the parameters & saves to output directory

    Parameters:
        corpus_path: the path to the corpus directory (os.path)
        output_path: the path to directory where index will be saved (os.path)
        model: the model to be used (Gensim.Model)
        name: the name of the index (str)
    """
    mc = MathCorpus(corpus_path)
    index = similarities.Similarity(output_path,
                                    model[mc], num_features=500)
    index.save(os.path.join(output_path, name + ".index"))


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
        create_model(self.corpus, self.output, lda=True, lsi=True, tfidf=True)
        path = os.path.join(self.output, "model.lda")
        self.lda = models.LdaModel.load(path)
        path = os.path.join(self.output, "model.lsi")
        self.lsi = models.LsiModel.load(path)
        path = os.path.join(self.output, "model.tfidf")
        self.tfidf = models.TfidfModel.load(path)

    def log(self, message):
        if self.debug:
            print(message)

    def tearDown(self):
        if os.path.exists(self.output):
            shutil.rmtree(self.output)

    def testLDA(self):
        create_index(self.corpus, self.output, self.lda, "test")
        index = similarities.Similarity.load(os.path.join(self.output,
                                                          "test.index"))
        p = "(stored under {})".format(str(self.output))
        expect = "Similarity index with 9 documents in 1 shards {}".format(p)
        self.assertEqual(expect, str(index))

    def testLSI(self):
        create_index(self.corpus, self.output, self.lsi, "test")
        index = similarities.Similarity.load(os.path.join(self.output,
                                                          "test.index"))
        p = "(stored under {})".format(str(self.output))
        expect = "Similarity index with 9 documents in 1 shards {}".format(p)
        self.assertEqual(expect, str(index))

    def testTFIDF(self):
        create_index(self.corpus, self.output, self.tfidf, "test")
        index = similarities.Similarity.load(os.path.join(self.output,
                                                          "test.index"))
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
    parser.add_argument("corpus",
                        help="The path to Math Corpus directory (html, xhtml)",
                        action="store")
    parser.add_argument("output",
                        help="The path to directory where model will be saved",
                        action="store")
    parser.add_argument("model",
                        help="The path to the model",
                        action="store")
    parser.add_argument("type",
                        help="The type of the model (1:tfidf, 2:lsi)",
                        type=int)
    parser.add_argument("name",
                        help="The name of the index")
    args = parser.parse_args()
    # need to load the model
    if args.type == 1:
        model = models.TfidfModel.load(args.model)
    elif args.type == 2:
        model = models.LsiModel.load(args.model)
    else:
        print("Unknown Model type")
        sys.exit()
    create_index(args.corpus, args.output, model, args.name)
