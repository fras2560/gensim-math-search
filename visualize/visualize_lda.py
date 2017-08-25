'''
Created on Aug 16, 2017

@author: d6fraser
'''
from gensim import models, corpora
import argparse
import pyLDAvis.gensim as gensimvis
import pyLDAvis
import os
import logging
logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)
logging.root.level = logging.INFO

if __name__ == "__main__":
    descp = """
            Visualize the created LDA model (see create_models.py)
            Author: Dallas Fraser (d6fraser@uwaterloo.ca)
            """
    parser = argparse.ArgumentParser(description=descp)
    prompt = "The path to Model directory (created by create_model)"
    parser.add_argument("model",
                        help=prompt,
                        action="store")
    args = parser.parse_args()
    model_path = args.model
    lda = models.LdaModel.load(os.path.join(model_path, "model.lda"))
    corpus = corpora.MmCorpus(os.path.join(model_path, "corpus.mm"))
    dictionary = corpora.Dictionary.load(os.path.join(model_path,
                                                      "corpus.dict"))
    print("Loaded")
    vis_data = gensimvis.prepare(lda, corpus, dictionary)
    print("prepared")
    pyLDAvis.display(vis_data)
    print("displayed")
