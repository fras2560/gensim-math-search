import unittest
import re
from bs4 import BeautifulSoup


def strip_tags(html):
    """Returns a string stripped of all html tags
    """
    words = re.sub(r'<.*?>', ' ', html)
    words = words.replace("  ", " ")
    return BeautifulSoup(words).text


class TestStriper(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def testStripTags(self):
        result = strip_tags("<html>Hello<html>")
        self.assertEqual(result.strip(), "Hello")
        result = strip_tags("<ol><li>Hello</li><li>baby</li></ol>")
        self.assertEqual(result.strip(), "Hello baby")

    def testMathDoc(self):
        test = """
                <div class="bibblock">
                   Willick, J.A., Courteau, S., Faber, S.M., Burstein, D.,
            Dekel, A., &amp; Strauss, M.A. 1997, ApJS, 109, 333
                            </div>"""
        result = strip_tags(test)
        expect = """Willick, J.A., Courteau, S., Faber, S.M., Burstein, D.,
      Dekel, A., & Strauss, M.A. 1997, ApJS, 109, 333"""
        self.assertEqual(result.strip(), expect)

if __name__ == "__main__":
    unittest.main()
