import time
import re

# from nltk.corpus import stopwords

docs = []
dictionary = {}

def index(filename):
    global docs
    global dictionary
    c = 0
    with open(filename) as f:

        t1 = time.time()
        for l in f:
            c += 1
            s = l.strip().lower().split("\t")
            words = s[2:-1] + s[-1].split(' ')[:-1]
            words[0] = words[0][1:]

            docs.append(l)
            doc_id = len(docs) - 1

            for w in words:
                w = re.sub('[#@]','',w)
                if len(w) > 1:
                    if w not in dictionary:
                        dictionary[w] = set()
                    dictionary[w].add(doc_id)

            docs.append(l)

            if c == 100000:
                break
        t2 = time.time()

        print(t2 - t1)
        print(c)

def query(s1,s2=''):
    match1 = dictionary.get(s1,[])
    match2 = None
    if s2 != '':
        match2 =  dictionary.get(s2,[])
        result = set(match1).intersection(match2)
    else:
        result = match1
    return result


index('tweets-1')





#test
t = query('my','car')

for v in t:
    print(docs[v])