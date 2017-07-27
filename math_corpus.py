from gensim import corpora
from six import iteritems
from htmlStriper import strip_tags
import os
import unittest
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from tangent.math_extractor import MathExtractor
from tangent.mathdocument import MathDocument

STOP_WORDS = set(stopwords.words('english'))


def convert_math_expression(mathml):
    """Returns the math tuples for a given math expression

    Parameters:
        mathml: the math expression (string)
    Returns:
        : a string of the math tuples
    """
    try:
        tokens = MathExtractor.math_tokens(mathml)
        pmml = MathExtractor.isolate_pmml(tokens[0])
        tree_root = MathExtractor.convert_to_mathsymbol(pmml)
        height = tree_root.get_height()
        eol = False
        if height <= 2:
            eol = True
        pairs = tree_root.get_pairs("", 1, eol=eol, unbounded=True)
        node_list = [format_node(node)
                     for node in pairs]
        return " ".join(node_list)
    except AttributeError:
        return ""


def format_node(node):
    """Returns a formatted node

    Parameters:
        node: the the math node (string)
    Returns:
        : a formatted node
    """
    node = str(node).lower()
    node = node.replace("*", "\*")
    for letter in "zxcvbnmasdfghjklqwertyuiop":
        node = node.replace("?" + letter, "*")
    return ((str(node)
            .replace(" ", "")
            .replace("&comma;", "comma")
            .replace("&lsqb;", "lsqb")
            .replace("&rsqb;", "rsqb")
             ))


def keep_word(word):
    """Returns true if the word should be kepts

    Parameters:
        word: the word to be checked (string)
    Returns:
        result: true if the word is worth keep (boolean)
    """
    result = False
    if len(word) > 1 and word not in STOP_WORDS and word.isalpha():
        result = True
    return result


def format_paragraph(paragraph, stemmer):
    """Returns a formatted paragraph

    Parameters:
        paragraph: the text paragraph to format (string)
        stemmer: the stemmer to use
    Returns:
        : a list of words (list)
    """
    result = strip_tags(paragraph)
    words = result.split(" ")
    return [stemmer.stem(word.lower().strip()) for word in words
            if keep_word(word.strip())]


class MathCorpus(object):
    def __init__(self, directory, filepath=None):
        """A gensim corpus for math documents uses tangent to encode MathMl

        Parameters:
            directory: the corpus directory (os.path)
            filepath: the filepath to the dictionary (os.path)
        """
        self.directory = directory
        if filepath is None:
            self.directory = directory
            fps = []
            for p, __, files in os.walk(self.directory):
                for file in files:
                    fps.append(os.path.join(p, file))
            self.length = len(fps)
            dictionary = corpora.Dictionary(self.parse_file(file)
                                            for file in fps)
            stop_ids = [dictionary.token2id[stopword]
                        for stopword in STOP_WORDS
                        if stopword in dictionary.token2id]
            once_ids = [tokenid
                        for tokenid, docfreq in iteritems(dictionary.dfs)
                        if docfreq == 1]
            # remove stop words and words that appear only once
            dictionary.filter_tokens(stop_ids + once_ids)
            # remove gaps in id sequence after words that were removed
            dictionary.compactify()
            self.dictionary = dictionary
        else:
            self.load_dictionary(filepath)
        print("Dictionary", self.dictionary)
        print(self.dictionary.token2id)

    def parse_file(self, filepath):
        """Returns the parsed contents of the file

        Parameters:
            filepath: the path to the file to be parsed (os.path)
        Returns:
            result: the parsed contents of the file (list)
        """
        if filepath.endswith(".xhtml") or filepath.endswith(".html"):
            result = ParseDocument(filepath).get_words().split(" ")
        else:
            result = []
        return result

    def load_dictionary(self, filepath):
        """Load a previous dictionary

        Parameters:
            filepath: the path to the dictionary (os.path)
        """
        self.dictionary = corpora.Dictionary.load(filepath)

    def save_dictionary(self, filepath):
        """Save the current dictionary

        Parameters:
            filepath: the path to where the dictionary will be saved (os.path)
        """
        self.dictionary.save(filepath)

    def __iter__(self):
        """ Yields one parsed document at a time"""
        for subdir, __, files in os.walk(self.directory):
            for file in files:
                filepath = os.path.join(subdir, file)
                if filepath.endswith(".xhtml") or filepath.endswith(".html"):
                    words = ParseDocument(filepath).get_words()
                    yield self.dictionary.doc2bow(words.split(" "))
                else:
                    # just skip for now
                    pass

    def __len__(self):
        """Returns the number of document in the corpus """
        return self.length


class ParseDocument(object):
    def __init__(self, filepath):
        """
        Parameters:
            filepath: the filepath to the document
        """
        self.filepath = filepath
        self.lines = []
        stemmer = PorterStemmer()
        (__, content) = MathDocument.read_doc_file(self.filepath)
        while len(content) != 0:
            (start, end) = MathExtractor.next_math_token(content)
            if start == -1:
                # can just print the rest
                self.lines.append(" ".join(format_paragraph(content, stemmer)))
                content = ""
            else:
                words = format_paragraph(content[0:start], stemmer)
                self.lines.append(" ".join(words))
                maths = convert_math_expression(content[start:end])
                self.lines.append(maths)
                # now move the content further along
                content = content[end:]

    def get_words(self):
        """Returns a string of the words parsed"""
        return " ".join(self.lines)


class TestMathCorpus(unittest.TestCase):
    def setUp(self):
        self.fp = os.path.join(os.getcwd(), "test")
        self.corpus = os.path.join(os.getcwd(), "tutorialDocuments")

    def tearDown(self):
        pass

    def test(self):
        mc = MathCorpus(self.fp)
        expected = [[(0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1)],
                    [(0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1)]]
        for index, vector in enumerate(mc):
            self.assertEqual(expected[index], vector)

    def testTutorial(self):
        mc = MathCorpus(self.corpus)
        expect = ['human', 'time', 'minor', 'comput', 'survey', 'user',
                  'system', 'interfac', 'respons', 'graph', 'tree', 'ep']
        for key in expect:
            self.assertEqual(key in mc.dictionary.token2id.keys(), True)


class TestMathDocument(unittest.TestCase):
    def setUp(self):
        self.fp = os.path.join(os.getcwd(), "test")

    def tearDown(self):
        pass

    def test1(self):
        md = ParseDocument(os.path.join(self.fp, "1.xhtml"))
        expect = ["mathemat",
                  "rigor",
                  "approach",
                  "quantum",
                  "field",
                  "theori",
                  "oper",
                  "algebra",
                  "in",
                  "case",
                  "('n!1','+','n')",
                  "('n!1','n!1')",
                  "('+','n!1','n')"]
        self.assertEqual(md.get_words().strip(), " ".join(expect))

    def test2(self):
        md = ParseDocument(os.path.join(self.fp, "2.xhtml"))
        expect = ["we",
                  "first",
                  "explain",
                  "formul",
                  "full",
                  "conform",
                  "theori",
                  "('n!1','+','n')",
                  "('n!1','n!1')",
                  "('+','n!1','n')",
                  "minkowski",
                  "algebra",
                  "quantum",
                  "field"]
        self.assertEqual(md.get_words().strip(), " ".join(expect))


class TestFunctions(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def testFormatParagraph(self):
        stemmer = PorterStemmer()
        result = format_paragraph("<h1> Hello</h1> <p>How are you</p>",
                                  stemmer)
        self.assertEqual(result, ['hello', 'how'])

    def testConverMathExpression(self):
        test = """
                <math display="inline" id="1_(number):0">
                  <semantics>
                    <mi mathvariant="normal">
                      I
                    </mi>
                    <annotation-xml encoding="MathML-Content">
                      <ci>
                        normal-I
                      </ci>
                    </annotation-xml>
                    <annotation encoding="application/x-tex">
                      \mathrm{I}
                    </annotation>
                  </semantics></math>
               """
        result = convert_math_expression(test)
        self.assertEqual(result, "('v!i','!0','n')")

    def testKeepWord(self):
        self.assertEqual(keep_word("they"), False)
        self.assertEqual(keep_word("hello"), True)
        self.assertEqual(keep_word("pep8"), False)

    def testFormatParagraph2(self):
        test = """
                <p>
                  There are two ways to write the real number 1 as a
                  <a href="recurring_decimal"
                  title="wikilink">recurring decimal</a>:
                  as 1.000..., and as
                  <a class="uri" href="0.999..." title="wikilink">0.999...</a>
                  (<em><a class="uri" href="q.v."
                  title="wikilink">q.v.</a></em>).
                  There is only one way to represent the real number 1
                  as a <a href="Dedekind_cut" title="wikilink">Dedekind cut</a>
                  <math display="block" id="1_(number):1">
                </p>
               """
        stemmer = PorterStemmer()
        result = format_paragraph(test, stemmer)
        expect = ['there', 'two', 'way', 'write', 'real', 'number',
                  'recur', 'decim', 'there', 'one', 'way',
                  'repres', 'real', 'number', 'dedekind', 'cut']
        self.assertEqual(result, expect)


if __name__ == "__main__":
    pass
