import numpy as np
import re
import math

documents = []
documents_raw = []
terms = {} # contains individual terms
count ,term_count = 0,0 # number of documents and terms

# Read Dataset and extract documents and all terms of the documents
with open('tweets-1', 'r') as f:
    #read and normalize tweets
    for line in f:
        count += 1
        line = line.strip().lower()
        words = line.split(' ')[5:] # remove metadata and seperate words
        documents_raw.append(' '.join(words))
        tf_count = dict()
        for w in words:
            if 'http' not in w: # remove links
                regex = re.compile('[^a-z]') # remove special characters
                w = regex.sub('',w)
                if len(w) > 0:
                    tf_count[w] = tf_count.get(w, 0) + 1  #count every term in document
        documents.append(tf_count)

        # count df value
        for c in tf_count:
            if c not in terms:
                terms[c] = [1, term_count] # term_count is the index in the tf-idf matrix
                term_count += 1
            else:
                terms[c][0] += 1
        # stop indexing
        if count % 1000 == 0:
            break

# idf score
for t in terms:
    terms[t][0] = math.log(count / terms[t][0])

# tf-idf score
for d in documents:
    for t in d:
        d[t] = (1 + math.log(d[t])) * terms[t][0]

# fill tf-idf  matrix
vectors = np.zeros((len(terms), count))
for i,d in enumerate(documents):
    for t in d:
        vectors[terms[t][1], i] = terms[t][0]

#normilize document vectors
norm = np.linalg.norm(vectors,axis=0)
norm[norm == 0] = 1 # avoid /0
for i in range(len(vectors)):
    vectors[i] = vectors[i] / norm

def query(a, b):
    return np.dot(a, b)

def print_similar_documents(q,d,n=100):
  scores = np.zeros((d.shape[1]))
  for i in range(d.shape[1]):
      scores[i] = query(q,vectors[:,i])
  order = np.argsort(scores)[::-1]

  print('query: ',documents_raw[order[0]])   # first is the query
  for o in order[1:n]:
      print(scores[o],documents_raw[o])


#### print similar tweets to given tweet
selected_tweet_index = 0
print_similar_documents(vectors[:,selected_tweet_index],vectors)

