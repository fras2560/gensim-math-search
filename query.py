'''
Name: Dallas Fraser
Email: d6fraser@uwaterloo.ca
Date: 2017-07-27
Project: Tangent GT
Purpose: To allow a user to query and run the NTCIR-MathIR task
'''
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
from math_corpus import convert_math_expression, format_paragraph
import sys
import os
import unittest
import argparse

PR_SCORE = 0
R_SCORE = 2.0


def reverse_lookup(dictionary, value):
    for key, v in dictionary.items():
        if value == v:
            return key


class Queries():
    """An Abstract class that shows what needs to be implemented
    to be used to check results for a specific benchmark

    To be Implemented:
        __init__(self, queries, results)
        test_indexer(self, indexer, output, top_k=10)
    """
    def __init__(self, queries, results):
        raise NotImplementedError("Constructor needs to be implemented")

    def test_indexer(self, indexer, output, top_k=10):
        """Outputs the score for each query

        Parameters:
            indexer: the indexer to use (Indexer)
            output: the path to the file to output to (path)
            top_k: the number of documents to retrieve
        """
        raise NotImplementedError("test_indexer needs to be implemented")


class Indexer():
    def __init__(self, dictionary, model, index, corpus_path):
        self.model = model
        self.dictionary = dictionary
        self.index = index
        self.collection = DocumentCollection(corpus_path)

    def search(self, query, top_k=10):
        """Returns the top k documents for the search

        Parameters:
            query: the query object to search (Query)
            top_k: how many results to return, default 10 (int)
        """
        # set the number of documents returned
        self.index.num_best = top_k
        # get the vec of the query
        vec_bow = self.dictionary.doc2bow(query.get_words().split(" "))
        vec_model = self.model[vec_bow]
        sims = self.index[vec_model]
        if (len(sims) > 0):
            if isinstance(sims[0], list) or isinstance(sims[0], tuple):
                sims = sorted(sims,
                              key=lambda item: -item[1])
            else:
                sims = sorted(enumerate(sims),
                              key=lambda item: -item[1])
        # build up the list of document names
        documents = []
        for doc in range(0, min(top_k, len(sims))):
            documents.append(self.collection.lookup(sims[doc][0]))
        return documents


class DocumentCollection():
    def __init__(self, corpus_path):
        self.mapping = {}
        index = 0
        for p, __, files in os.walk(corpus_path):
            for file in files:
                self.mapping[index] = os.path.join(p, file)
                index += 1

    def lookup(self, index):
        """Returns the document that maps to the index

        Parameters:
            index: the index of the document
        """
        document = None
        try:
            document = self.mapping[index]
        except KeyError:
            pass
        return document


class ArxivQueries(Queries):
    def __init__(self, queries, results):
        """ArxivQueries: the queries for arxiv task NTCIR MathIR 12

        Parameters:
            queries: the queries file (os.path)
            results: the results file (os.path)
        """
        with open(queries) as doc:
            self.soup = BeautifulSoup(doc)
            self.queries = []
            for topic in self.soup.find_all("topic"):
                self.queries.append(Query(topic))
        with open(results) as doc:
            self.results = ExpectedResults(doc)

    def test_indexer(self, indexer, output, top_k=10):
        """Outputs the score for each query

        Parameters:
            indexer: the indexer to use (Indexer)
            output: the path to the file to output to (path)
            top_k: the number of documents to retrieve
        """
        with open(output, "w+") as out_doc:
            for query in self.queries:
                r_docs = 0
                pr_docs = 0
                results = indexer.search(query, top_k=top_k)
                print(results)
                for result in results:
                    score = self.results.find_score(query, result)
                    if score > PR_SCORE:
                        pr_docs += 1
                    if score > R_SCORE:
                        r_docs += 1
                print("{},{},{}".format(query, r_docs, pr_docs),
                      file=out_doc)


class ExpectedResults():
    def __init__(self, document):
        """ExpectedResults: used to find the scores
                            for a document for a given query

        Parameters:
            document: the document with the queries and results
        """
        self.results = {}
        for line in document:
            parts = line.split(" ")
            if len(parts) == 4:
                query_name = parts[0]
                document_name = parts[2]
                score = parts[3]
                if query_name not in self.results.keys():
                    self.results[query_name] = {}
                self.results[query_name][document_name] = float(score.strip())

    def find_score(self, query, document):
        """Returns the score for the document for a given query

        Parameters:
            query: the query object (Query)
            document: the name of the document (str)
        Returns:
            result: the resulting score, default -1 (int)
        """
        result = -1
        try:
            result = self.results[str(query)][self.parse_name(document)]
        except KeyError:
            pass
        return result

    def parse_name(self, document):
        """Returns a parse document name

        Parameters:
            document: the document name to parse (str)
        Returns:
            filename: the parse filename (str)
        """
        path, filename = os.path.split(document)
        if ".xml" in filename or ".html" in filename or "xhtml" in filename:
            filename = ".".join(filename.split(".")[0:-1])
        return filename


class Query():
    def __init__(self, topic):
        """Query: the NTCIR-MathIR query

        Parameters:
            topic: the soup topic object (bs4)
        """
        stemmer = PorterStemmer()
        keywords = []
        for keyword in topic.find_all("keyword"):
            keywords.append(" ".join(format_paragraph(keyword.text,
                                                      stemmer)))
        formulas = []
        for formula in topic.find_all("formula"):
            formulas.append(convert_math_expression(str(formula)))
        self.result = keywords + formulas
        self.name = topic.num.text
        self.result = [result for result in self.result
                       if result != ""]

    def get_words(self):
        """Returns the clauses of the query (str)
        """
        return " ".join(self.result)

    def __str__(self):
        """Returns the name of the query (str)
        """
        return self.name


if __name__ == "__main__":
    descp = """
            Test the Gensim index(s) created by create_index.py
            Author: Dallas Fraser (d6fraser@uwaterloo.ca)
            """
    parser = argparse.ArgumentParser(description=descp)
    parser.add_argument('-lsi',
                        dest="lsi",
                        action="store_true",
                        default=False,
                        help="Search with LSI Index")
    parser.add_argument('-lda',
                        dest="lda",
                        action="store_true",
                        help="Search with LDA Index",
                        default=False)
    parser.add_argument('-tfidf',
                        dest="tfidf",
                        action="store_true",
                        help="Search with TFIDF Index",
                        default=False)
    parser.add_argument('-hdp',
                        dest="hdp",
                        action="store_true",
                        help="Search with HDP Index",
                        default=False)
    prompt = "The path to Index directory (created by create_index)"
    parser.add_argument("index",
                        help=prompt,
                        action="store")
    prompt = "The name of the Index"
    parser.add_argument("index_name",
                        help=prompt,
                        action="store")
    prompt = "The path to Model directory (created by create_models)"
    parser.add_argument("model",
                        help=prompt,
                        action="store")
    args = parser.parse_args()
