'''
Name: Dallas Fraser
Email: d6fraser@uwaterloo.ca
Date: 2017-07-27
Project: Tangent GT
Purpose: To test the created models on the NTCIR-MathIR Arxiv task
'''
import sys
sys.path.append("..") # Adds higher directory to python modules path.
from ranking.query import Indexer, ArxivQueries
import argparse
import os

if __name__ == "__main__":
    descp = """
            Test the Gensim index(s) created by create_index.py
            Author: Dallas Fraser (d6fraser@uwaterloo.ca)
            """
    parser = argparse.ArgumentParser(description=descp)
    prompt = "The path to Index Folder (created by create_index)"
    parser.add_argument("index",
                        help=prompt,
                        action="store")
    parser.add_argument('top_k', default=10, type=int,
                        help='The number of results for each search',
                        nargs='?')
    prompt = "The path to Models' directory (created by create_models)"
    parser.add_argument("models",
                        help=prompt,
                        action="store")
    parser.add_argument("corpus",
                        help="The path to Math Corpus directory (html, xhtml)",
                        action="store")
    parser.add_argument("output",
                        help="The path to output directory",
                        action="store")
    parser.add_argument("queries",
                        help="The path to queries file",
                        action="store")
    parser.add_argument("results",
                        help="The path to results file",
                        action="store")
    args = parser.parse_args()
    aq = ArxivQueries(args.queries, args.results)
    index = Indexer(args.models, args.index, args.corpus)
    aq.test_indexer(index, os.path.join(args.output, "result"), args.top_k)
