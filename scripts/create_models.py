'''
Name: Dallas Fraser
Email: d6fraser@uwaterloo.ca
Date: 2017-07-27
Project: Tangent GT
Purpose: To create the models to be used for indexing
'''
import sys
sys.path.append("..") # Adds higher directory to python modules path.
from tangent.math_corpus import MathCorpus
from gensim import models, corpora
import os
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
    parser.add_argument('num_topics', default=500, type=int,
                        help='The number of results for each search',
                        nargs='?')
    parser.add_argument("corpus",
                        help="The path to Math Corpus directory (html, xhtml)",
                        action="store")
    parser.add_argument("output",
                        help="The path to directory where model will be saved",
                        action="store")
    args = parser.parse_args()
    create_model(args.corpus,
                 args.output,
                 num_topics=args.num_topics,
                 tfidf=args.tfidf,
                 lda=args.lda,
                 lsi=args.lsi,
                 hdp=args.hdp)
