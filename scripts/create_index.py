'''
Name: Dallas Fraser
Email: d6fraser@uwaterloo.ca
Date: 2017-07-27
Project: Tangent GT
Purpose: To create the indexes used for querying and similarity
'''
import sys
sys.path.append("..") # Adds higher directory to python modules path.
from gensim import similarities, models, corpora
import os
import argparse
import logging
logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)
logging.root.level = logging.INFO


def create_index(corpus_path,
                 output_path,
                 model_path,
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
    tfidf_model = models.TfidfModel.load(os.path.join(model_path,
                                                      "model.tfidf"))
    if tfidf:
        op = os.path.join(output_path, "tfidf")
        index = similarities.Similarity(op,
                                        tfidf_model[mc],
                                        num_features=len(dictionary))
        index.save(os.path.join(output_path, "index.tfidf"))
    if lda:
        model = models.LdaModel.load(os.path.join(model_path, "model.lda"))
        op = os.path.join(output_path, "lda")
        index = similarities.Similarity(op,
                                        model[mc],
                                        num_features=model.num_topics)
        index.save(os.path.join(output_path, "index.lda"))
    if lsi:
        model = models.LsiModel.load(os.path.join(model_path, "model.lsi"))
        op = os.path.join(output_path, "lsi")
        index = similarities.Similarity(op,
                                        model[tfidf_model[mc]],
                                        num_features=model.num_topics)
        index.save(os.path.join(output_path, "index.lsi"))
    if hdp:
        model = models.HdpModel.load(os.path.join(model_path, "model.hdp"))
        op = os.path.join(output_path, "hdp")
        index = similarities.Similarity(op,
                                        model[mc],
                                        num_features=model.m_T)
        index.save(os.path.join(output_path, "index.hdp"))


class ModelException(Exception):
    pass


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
