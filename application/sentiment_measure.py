import logging
import math
import logging as log

import numpy

from activation import none, tanh, sig, relu
from network import Network

# labels
# scores_path = "data/labels_test.txt"
# reviews_path = "data/reviews_test.txt"

scores_path = "data/labels.txt"
reviews_path = "data/reviews.txt"

scores = []
with open(scores_path, "r") as file:
    for line in file:
        scores.append(line)

reviews = []
with open(reviews_path, "r") as file:
    for line in file:
        reviews.append(line)

reviews_count = len(reviews)

assert len(scores) == reviews_count

log.info(f'Reviews nr: {reviews_count}')

reviews_scores = []

for i in range(reviews_count):
    reviews_scores.append((reviews[i], scores[i]))

assert len(reviews_scores) == reviews_count

# create vocabulary

vocabulary_raw = list()
vocabulary = dict()

for review in reviews:
    vocabulary_raw.extend(review.split(" "))

log.info(f'Created raw set of {len(vocabulary_raw)} words')

for item in vocabulary_raw:
    if len(item.strip()) > 1:
        if item.strip() not in vocabulary:
            vocabulary[item.strip()] = 0
        else:
            vocabulary[item.strip()] = vocabulary[item.strip()] + 1

log.info(f'Created cleaned map of {len(vocabulary.keys())}')

count0 = sum(1 for value in vocabulary.values() if value > 0)
count5 = sum(1 for value in vocabulary.values() if value >= 5)
count10 = sum(1 for value in vocabulary.values() if value >= 10)
count20 = sum(1 for value in vocabulary.values() if value >= 20)
count50 = sum(1 for value in vocabulary.values() if value >= 50)
count100 = sum(1 for value in vocabulary.values() if value >= 100)
count500 = sum(1 for value in vocabulary.values() if value >= 500)
count700 = sum(1 for value in vocabulary.values() if value >= 700)
count1000 = sum(1 for value in vocabulary.values() if value >= 1000)
count2000 = sum(1 for value in vocabulary.values() if value >= 2000)

log.info(f'Nr of words with any incidence: {count0}')
log.info(f'Nr of words with incidence >= 5: {count5}')
log.info(f'Nr of words with incidence >= 10: {count10}')
log.info(f'Nr of words with incidence >= 20: {count20}')
log.info(f'Nr of words with incidence >= 50: {count50}')
log.info(f'Nr of words with incidence >= 100: {count100}')
log.info(f'Nr of words with incidence >= 500: {count500}')
log.info(f'Nr of words with incidence >= 700: {count700}')
log.info(f'Nr of words with incidence >= 1000: {count1000}')
log.info(f'Nr of words with incidence >= 2000: {count2000}')

vocabulary_list = list(key for key, value in vocabulary.items() if value >= 2000)

vocabulary_size = len(vocabulary_list)

log.info(f'Nr of words in vocabulary: {vocabulary_size}')

word_index = dict()

for i, voc_entry in enumerate(vocabulary_list):
    word_index[voc_entry] = i

review_vectors = []

# create review vector

for review in reviews:
    review_vector = numpy.zeros(vocabulary_size)
    review_words = review.split(" ")

    for word in review_words:
        if word.strip() in word_index:
            review_vector[word_index[word.strip()]] = 1

    review_vectors.append(review_vector)

# log.info(review_vectors[0])

assert len(review_vectors) == reviews_count

log.info(f'Nr of words in training set: {reviews_count}')
log.info(f'Datapoint length: {len(review_vectors[0])}')


def map_score(label):
    if label.strip() == 'positive':
        return [1]
    if label.strip() == 'negative':
        return [0]


log.info('Mapped scores')

# create scores vector
scores = list(map(lambda p: map_score(p), scores))

assert len(review_vectors) == len(scores)

# select training data

training_reviews = review_vectors[0:2000]
training_scores = scores[0:2000]

# training_reviews = review_vectors[0:3000]
# training_scores = scores[0:3000]

log.info('Created training data')

# select test data
test_reviews = review_vectors[24000:25000]
test_scores = scores[24000:25000]

log.info('Created test data')

# create network

layers = [vocabulary_size, 128, 1]
dropout = [0.3, 0]
network = Network(layers, dropout, [sig, sig], False)

log.info(f'test positives: {test_scores.count([1])}')
log.info(f'test negatives: {test_scores.count([0])}')

log.info('Network ready')

log.info(f'Train reviews: {len(training_reviews)}')
log.info(f'Train scores: {len(training_scores)}')

iterations = len(training_reviews)


# for i in range(iterations):
#     network.learn(training_reviews[i], training_scores[i])
#     log.info(f'Training iteration {iter_nr} of {iterations}')
#     iter_nr += 1

def learn_epoch(epoch_nr):
    iter_nr = 0

    for i in range(iterations):
        network.learn(training_reviews[i], training_scores[i])
        log.info(f'Training iteration {iter_nr} of {iterations}, epoch {epoch_nr}')
        iter_nr += 1

    log.info('Learning finished')

    verify = []

    for i in range(len(test_reviews)):
        predicted = network.predict(test_reviews[i], False)
        real = test_scores[i]
        log.info(f'Prediction: {predicted}')
        log.info(f'Actual: {test_scores[i]}')

        if (real == [1] and predicted[0] > 0.50) or (real == [0] and predicted[0] < 0.50):
            verify.append(1)
        else:
            verify.append(-1)

    log.info('--------------------------------------------------------')
    log.info(f'Testing finished, epoch {epoch_nr} \n')
    log.info('SUMARY:')
    log.info(f'Total learnig cases: {len(training_reviews)}')
    log.info(f'Total testing cases: {len(test_reviews)}')
    log.info(f'Total testing scores: {len(verify)}')
    log.info(f'Total success predictions: {verify.count(1)}')
    log.info(f'Total failed predictions: {verify.count(-1)}')
    log.info(f'Total uncertain predictions: {verify.count(0)}')
    log.info(f'Success rate: {verify.count(1) / len(verify)}')
    log.info(f'Fail rate: {verify.count(-1) / len(verify)}')
    log.info('-------------------------------------------------------')


for i in range(1, 2):
    learn_epoch(i)

# TESTING SETUP, EXPECTED:
# --------------------------------------------------------
# Testing finished, epoch 1
# SUMARY:
# Total learnig cases: 2000
# Total testing cases: 1000
# Total testing scores: 1000
# Total success predictions: 720
# Total failed predictions: 280
# Total uncertain predictions: 0
# Success rate: 0.72
# Fail rate: 0.28
# Uncertain rate: 0.0
# -------------------------------------------------------
#
#
#
#
#
#



