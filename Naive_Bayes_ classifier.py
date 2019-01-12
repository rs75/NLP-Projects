import pandas
import numpy as np
from nltk.corpus import stopwords

pandas.set_option('display.width', 1000)
stop = set(stopwords.words('german'))

count_words_in_class = [0, 0]
p_tc = [{}, {}]  # for both classes
p = [0, 0]  # probability that documents is in c1/c2


def load_file(filename, train=False):
    def count_words(text, c):
        for w in text:
            p_tc[c][w] = p_tc[c].get(w, 0) + 1  # count occurance of terms in both classes
        count_words_in_class[c] += 1

    # import csv file and remove appname
    data = pandas.read_csv(filename, delimiter='\t', header=None, names=['class', 'title', 'description'], usecols=range(1, 4))
    data = data.replace(np.nan, '', regex=True)

    # convert gut/schlecht to boolean
    data.loc[data['class'] == 'gut', 'class'] = 1
    data.loc[data['class'] == 'schlecht', 'class'] = 0
    data['class'].astype(bool)

    # preprocess text and combine title and description
    data['text'] = data['title'] + " " + data['description']
    data = data.drop(['title', 'description'], axis=1)
    data['text'] = data['text'].apply(lambda x: x.strip().lower().replace('.', '').replace(',', '').split(" "))
    data['text'] = data['text'].apply(lambda x: [y for y in x if y not in stop])

    if train:
        for row in data.loc[data['class'] == True, 'text']:
            count_words(row, 1)
        for row in data.loc[data['class'] == False, 'text']:
            count_words(row, 0)

        alle = len(data)
        gut = np.sum(data['class'] == True)
        schlecht = alle - gut
        p[0], p[1] = schlecht / alle, gut / alle

        # compute p_tc
        for i, d in enumerate(p_tc):
            for k in d:
                p_tc[i][k] = (p_tc[i][k] + 1) / (count_words_in_class[i] + len(d))
    return data


def predict_class(text):
    class_prediction = [0, 0]
    for c in [0, 1]:
        temp = 1
        for w in text:
            if w in p_tc[c]:
                temp *= p_tc[c][w]
            else:
                temp *= 1 / (count_words_in_class[c] + len(p_tc[c]))
        class_prediction[c] = temp# + p[c] # better withour + p[c]
        #print(p[c],temp)
    return 1 if class_prediction[0] < class_prediction[1] else 0


def print_terms_with_height_propability(n=100, c=1):
    s = sorted(p_tc[c], key=p_tc[c].get, reverse=True)
    print('Top {}  words in class good'.format(n), s[:n])


data_train = load_file('games-train.csv', train=True)
data_test = load_file('games-test.csv', train=True)
data_test['predicted_class'] = data_test['text'].apply(lambda x: predict_class(x))

correct_classes = data_test['class'].values
predicted_classes = data_test['predicted_class'].values


def evaluate():
    # predict class good
    TP = np.sum((predicted_classes == 1) & (correct_classes == 1))
    FP = np.sum((predicted_classes == 1) & (correct_classes == 0))
    FN = np.sum((predicted_classes == 0) & (correct_classes == 1))
    P = TP / (TP + FP)
    R = TP / (TP + FN)
    print('precion class good: ', P)
    print('recall class good: ', R)
    print('F1 class good: ', (2 * P * R) / (P + R))
    print('TP: {}, FP: {}, FN:{}'.format(TP,FP,FN))

    # predict class bad
    TP = np.sum((predicted_classes == 0) & (correct_classes == 0))
    FP = np.sum((predicted_classes == 0) & (correct_classes == 1))
    FN = np.sum((predicted_classes == 1) & (correct_classes == 0))

    P = TP / (TP + FP)
    R = TP / (TP + FN)
    print('precion class bad: ', P)
    print('recall class bad: ', R)
    print('F1 class bad: ', (2 * P * R) / (P + R))
    print('TP: {}, FP: {}, FN: {}'.format(TP,FP,FN))


print_terms_with_height_propability()
evaluate()
