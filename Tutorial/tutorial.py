from gensim import corpora, models, similarities
from six import iteritems
import logging
logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)
logging.root.level = logging.INFO  # ipython sometimes messes up the logging setup; restore
# tutorial part 1
def reverse_lookup(dictionary, value):
	for key, v in dictionary.items():
		if value == v:
			return key
stoplist = set('for a of the and to in'.split())
# collect statistics about all tokens
dictionary = corpora.Dictionary(line.lower().split() for line in open('document.txt'))
# remove stop words and words that appear only once
stop_ids = [dictionary.token2id[stopword] for stopword in stoplist
            if stopword in dictionary.token2id]
once_ids = [tokenid for tokenid, docfreq in iteritems(dictionary.dfs) if docfreq == 1]
dictionary.filter_tokens(stop_ids + once_ids)  # remove stop words and words that appear only once
dictionary.compactify()  # remove gaps in id sequence after words that were removed
# remove common words and tokenize
class MyCorpus(object):
    def __iter__(self):
        for line in open('document.txt'):
	        # assume there's one document per line, tokens separated by whitespace
            yield dictionary.doc2bow(line.lower().split())

    def __len__(self):
    	return 9

print(dictionary)
print(dictionary.token2id)
print(dictionary.dfs)
for did, count in dictionary.dfs.items():
	print(reverse_lookup(dictionary.token2id, did), count)

corpus = MyCorpus()
tfidf = models.TfidfModel(corpus)
corpus_tfidf = tfidf[corpus]
lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=2)
lsi.print_topics(2)
for i in lsi.print_topics():
    print(i)

doc = "Human computer interaction"
vec_bow = dictionary.doc2bow(doc.lower().split())
print(vec_bow)
vec_lsi = lsi[vec_bow] # convert the query to LSI space
print(vec_lsi)
index = similarities.MatrixSimilarity(lsi[corpus])
sims = index[vec_lsi]
sims = sorted(enumerate(sims), key=lambda item: -item[1])
print(sims)
