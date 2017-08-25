'''
Name: Dallas Fraser
Email: d6fraser@uwaterloo.ca
Date: 2017-07-27
Project: Tangent GT
Purpose: Tests ranking objects and functions
'''
from ranking import compare_math_formula, Query, Document
from tangent.math_corpus import ParseDocument
from bs4 import BeautifulSoup
import unittest
import os


class TestFunctions(unittest.TestCase):

    def setUp(self):
        self.path = os.path.join(os.getcwd(),
                                 "test",
                                 "mathDocuments",
                                 "test.html")
        self.doc = ParseDocument(self.path)

    def tearDown(self):
        pass

    def testCompareMathFormula(self):
        math = self.doc.get_math()
        formula = math[0].split(" ")[0]
        query = "('n!1','+','n')"
        self.assertEqual(True, compare_math_formula(query, formula))
        query = "('n!1','*','n')"
        self.assertEqual(True, compare_math_formula(query, formula))
        query = "('n!1','\*','n')"
        self.assertEqual(False, compare_math_formula(query, formula))
        query = "('n!1','*','x')"
        self.assertEqual(False, compare_math_formula(query, formula))
        query = "('n!2','*','n')"
        self.assertEqual(False, compare_math_formula(query, formula))


class TestQuery(unittest.TestCase):
    def setUp(self):
        self.path = os.path.join(os.getcwd(),
                                 "test",
                                 "Tutorial",
                                 "testQueries.html")

    def tearDown(self):
        pass

    def testConstructor(self):
        expect = ["test-1: + human comput interact",
                  "test-2: + tree order"]
        with open(self.path) as doc:
            self.soup = BeautifulSoup(doc)
            self.queries = {}
            index = 0
            for topic in self.soup.find_all("topic"):
                query = Query(topic)
                self.assertEqual(str(query), expect[index])
                index += 1


class TestDocument(unittest.TestCase):
    def setUp(self):
        self.path = os.path.join(os.getcwd(),
                                 "test",
                                 "mathDocuments",
                                 "test.html")
        self.query_path = os.path.join(os.getcwd(),
                                       "test",
                                       "mathDocuments",
                                       "testQuery.html")
        with open(self.query_path) as doc:
            self.soup = BeautifulSoup(doc)
            self.queries = {}
            for topic in self.soup.find_all("topic"):
                self.query = Query(topic)

        self.doc = Document(self.path)

    def tearDown(self):
        pass

    def testCalculateTextWeights(self):
        result = self.doc.calculate_text_weights(self.query)
        expect = (2, ['neumann', 'neumann'], 2.0)
        self.assertEqual(expect, result)

    def testCalculateMathWeights(self):
        result = self.doc.calculate_math_weights(self.query)
        expect = ("('n!1','+','n') ('n!1','n!1') ('+','n!1','n')",
                  3,
                  ["('n!1','+','n')", "('n!1','n!1')", "('+','n!1','n')"],
                  1.0)
        result = result[0]
        self.assertEqual(result[0], expect[0])
        self.assertEqual(result[1], expect[1])
        self.assertEqual(result[2], expect[2])
        self.assertAlmostEqual(result[3], expect[3])

    def testMatchingParts(self):
        print(self.doc.words, self.doc.formulas)
        # test the same formula against itself
        query = ["('n!1','+','n')", "('n!1','n!1')", "('+','n!1','n')"]
        formulas = [
                    ["('n!1','+','n')", "('n!1','n!1')", "('+','n!1','n')"],
                    ["('v!s','n!1','a')", "('n!1','!0','n')"]
                   ]
        result = self.doc.matching_parts(query, formulas[0])
        expect = (3, ["('n!1','+','n')", "('n!1','n!1')", "('+','n!1','n')"])
        self.assertEqual(result, expect)
        # query different from index
        query = ["('n!2','+','n')", "('n!2','n!1')", "('+','n!1','n')"]
        result = self.doc.matching_parts(query, formulas[0])
        expect = (1, ["('+','n!1','n')"])
        self.assertEqual(result, expect)
        # formula larger than query
        query = ["('n!1','+','n')", "('+','n!1','n')"]
        result = self.doc.matching_parts(query, formulas[0])
        expect = (2, ["('n!1','+','n')", "('+','n!1','n')"])
        self.assertEqual(result, expect)
        # query has wild card
        query = ["('n!1','*','n')", "('*','n!1')", "('+','n!1','n')"]
        result = self.doc.matching_parts(query, formulas[0])
        expect = (3, ["('n!1','+','n')", "('n!1','n!1')", "('+','n!1','n')"])
        self.assertEqual(result, expect)


class TestRanking(unittest.TestCase):
    pass
if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
