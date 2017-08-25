'''
Name: Dallas Fraser
Email: d6fraser@uwaterloo.ca
Date: 2017-07-27
Project: Tangent GT
Purpose: Contains a ranking object used to rank a query
'''
from nltk.stem.porter import PorterStemmer
from bs4 import BeautifulSoup
from tangent.math_corpus import convert_math_expression, format_paragraph,\
                                ParseDocument
from gensim import corpora, models, similarities
from gensim.matutils import cossim
from ranking.metrics import jensen_shannon_divergence, hellinger_distance
import os


def compare_math_formula(query, formula):
    """Compares two math tuples

    Parameters:
        query: the query math tuple (str)
        formula: the formula math tuple (str)
    Returns:
        same: True if tuples are considered equal (boolean)
    """
    if "'*'" in query:
        # break on the wild card
        query_parts = query.split("'*'")
        index = 0
        # make sure all the parts are in the formula
        while index < len(query_parts) and query_parts[index] in formula:
            index += 1
        same = False
        # if all parts of query are in formula then must be good to go
        if index == len(query_parts):
            same = True
    else:
        # if no wildcards then just a normal str comparison
        same = query == formula
    return same


class Query():
    """Query Object used by ranking
    """
    def __init__(self, topic):
        """Query Object

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
            form = convert_math_expression(str(formula))
            formulas.append(form)
        self.formulas = formulas
        self.keywords = keywords
        self.name = topic.num.text

    def __str__(self):
        return ("{}:{} + {}".format(self.name,
                                    " ".join(self.formulas),
                                    " ".join(self.keywords)))


class Document():
    """Document Object used for Ranking
    """
    def __init__(self, filepath):
        """

        Parameters:
            filepath: the path to the file of the document (os.path)
        """
        self.document = ParseDocument(filepath)
        self.formulas = self.document.get_math()
        self.words = self.document.get_text()

    def calculate_text_weights(self, query):
        """Returns a list of weights of the text matched

        Parameters:
            query: a ranking query object (Query)
        Returns:
            weights: a tuple (numbers of terms matched,
                              matching terms,
                              % of terms)
        """
        matched = []
        for word in self.words:
            if word in query.keywords:
                matched.append(word)
        return (len(matched),
                matched,
                float(len(matched)) / len(query.keywords))

    def calculate_math_weights(self, query):
        """Returns a list of weights of the math formula matched

        Parameters:
            query: a ranking query object (Query)
        Returns:
            weights: a list of tuples (formula,
                                       number of terms matched,
                                       matching terms,
                                       % of terms,
                                       )
        """
        weights = []
        for query_formula in query.formulas:
            max_matched = 0
            max_matches = []
            for check in self.formulas:
                fm = query_formula.split(" ")
                (num_matched, matched) = self.matching_parts(fm,
                                                             check.split(" "))
                # check to see if this formula is a better match
                if num_matched > max_matched:
                    max_matched = num_matched
                    max_matches = matched
                    count = len(check.split(" "))
            weights.append((query_formula,
                            max_matched,
                            max_matches,
                            float(max_matched) / float(count)))
        return weights

    def matching_parts(self, query, formula):
        """Returns the intersection of the query and the formula

        Parameters:
            query: the math query (list of math tuples)
            formula: the math formula (list of math tuples)
        """
        matched = []
        query_index = 0
        formula_index = 0
        last_formula_match = formula_index
        while query_index < len(query):

            if compare_math_formula(query[query_index],
                                    formula[formula_index]):
                query_index += 1
                last_formula_match = formula_index
                matched.append(formula[formula_index])
            formula_index += 1
            if formula_index >= len(formula):
                formula_index = last_formula_match
                query_index += 1
        return (len(matched), matched)


class Ranking():
    def __init__(self,
                 query_document,
                 results,
                 prefix=""):
        """Ranking Object

        Parameters:
            query_document: a document that holds all the queries (os.file)
            results: a document that holds all the results (os.file)
            prefix: if the results documents have a prefix (os.path)
        """
        with open(query_document) as doc:
            self.soup = BeautifulSoup(doc)
            self.queries = {}
            for topic in self.soup.find_all("topic"):
                query = Query(topic)
                self.queries[query.name] = query
        with open(results) as doc:
            self.documents = {}
            for line in doc:
                parts = line.split(" ")
                query_name = parts[0]
                document_name = parts[2]
                if query_name not in self.results.keys():
                    self.results[query_name] = {}
                dname = os.path.join(prefix, document_name)
                self.documents[query_name].append(dname)

    def mean(self, numbers, index):
        """Returns the means of a list of tuples using the index

        Parameters:
            numbers: a list of tuples
            index: the index to use for the tuples
        """
        return float(sum([item[index] for item in numbers]) /
                     max(len(numbers), 1))

    def calculate_metrics(self, models, index, output):
        """Returns a dictionary of metrics for each document

        Parameters:
            models: directory that holds the models and dictionary (os.path)
            indexes: the directory holding the indexes (os.path)
            output: the os path where to output the results (os.path)
        Outputs:
            : outputs to the file
        """
        # load the various models
        self.tfidf_model = models.TfidfModel.load(os.path.join(models,
                                                               "model.tfidf"))
        self.lsi_model = models.LsiModel.load(os.path.join(models,
                                                           "model.lsi"))
        self.lda_model = models.LdaModel.load(os.path.join(models,
                                                           "model.lda"))
        self.dictionary = corpora.Dictionary.load(os.path.join(models,
                                                               "corpus.dict"))
        with open(output, "w+") as results:
            print("query,document,mmf%,amf%,#mtm,#ttm,mm,tm,lda,lsi,tfidf",
                  file=results)
            for query_name, query in self.queries:
                query_bow = self.dictionary.doc2bow(query
                                                    .get_words()
                                                    .split(" "))
                query_lda = self.lda_model(query_bow)
                query_tfidf = self.tfidf_model(query_bow)
                query_lsi = self.lsi_model(query_tfidf)
                for document in self.documents[query_name]:
                    doc = Document(document)
                    text_weights = doc.calculate_text_weights(query)
                    math_weights = doc.calculate_math_weights(query)
                    mmf = max(math_weights, key=lambda item: [-1])[-1]
                    amf = self.mean(math_weights, -1)
                    mtm = sum(item[1] for item in math_weights)
                    ttm = len(text_weights[0])
                    mm = True if mtm > 0 else False
                    tm = True if ttm > 0 else False
                    l = "{},{},{:.2f},{},{},"
                    r = "{},{:.2f},{:.2f},{:.2f}"
                    # what the document terms
                    doc_bow = self.dictionary.doc2bow(document
                                                      .get_words()
                                                      .split(" "))
                    tfidf_vec = self.tfidf_model[doc_bow]
                    lsi_vec = self.lsi_model[tfidf_vec]
                    lda_vec = self.lda_model[doc_bow]
                    tfidf = gensim.matutils.cossim(query_tfidf, tfidf_vec)
                    lsi = gensim.matutils.cossim(query_lsi, lsi_vec)
                    lda = jensen_shannon_divergence(query_lda,
                                                    lda_vec,
                                                    self.lda_model)
                    s = (l + r).format(query_name,
                                       document,
                                       mmf,
                                       amf,
                                       mtm,
                                       mm,
                                       tm,
                                       lda,
                                       lsi,
                                       tfidf
                                       )
                    print(s, file=results)
