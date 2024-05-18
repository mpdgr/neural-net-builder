import json
import logging as log
import time
from datetime import datetime
import jsonpickle

from train_summary import *

from activation import none, tanh, sig, relu
from network import Network

# labels
# scores_path = "data/labels_test.txt"
# reviews_path = "data/reviews_test.txt"
log.getLogger().setLevel(log.INFO)

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

vocabulary_list = list(key for key, value in vocabulary.items() if value >= 700)

vocabulary_size = len(vocabulary_list)

log.info(f'Nr of words in vocabulary: {vocabulary_size}')

word_index = dict()

for i, voc_entry in enumerate(vocabulary_list):
    word_index[voc_entry] = i

review_vectors = []

# create review vector

for review in reviews:
    review_vector = [0] * vocabulary_size
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

training_reviews = review_vectors[0:24000]
training_scores = scores[0:24000]

# training_reviews = review_vectors[0:3000]
# training_scores = scores[0:3000]

log.info('Created training data')

# select test data
test_reviews = review_vectors[24000:25000]
test_scores = scores[24000:25000]

log.info('Created test data')

# create network

layers = [vocabulary_size, 512, 1]
dropout = [0, 0.3]
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

    start_time = int(time.time())

    for i in range(iterations):
        network.learn(training_reviews[i], training_scores[i])
        log.info(f'Training iteration {iter_nr} of {iterations}, epoch {epoch_nr}')
        iter_nr += 1

    finish_time = int(time.time())
    total_time_sec = finish_time - start_time

    log.info(f'Learning finished, epoch learning time: {total_time_sec} seconds')


    verify = []

    confusion_matrix = {'true positive': 0, 'true negative': 0, 'false positive': 0, 'false negative': 0}

    for i in range(len(test_reviews)):
        predicted = network.predict(test_reviews[i], False)
        real = test_scores[i]
        log.info(f'Prediction: {predicted}')
        log.info(f'Actual: {test_scores[i]}')

        if (real == [1] and predicted[0] >= 0.50):
            verify.append(1)
            confusion_matrix['true positive'] += 1
        elif (real == [0] and predicted[0] < 0.50):
            verify.append(1)
            confusion_matrix['true negative'] += 1
        elif (real == [1] and predicted[0] < 0.50):
            verify.append(-1)
            confusion_matrix['false positive'] += 1
        elif (real == [0] and predicted[0] >= 0.50):
            verify.append(-1)
            confusion_matrix['false negative'] += 1

    summary = TrainSummary()
    summary.epoch_nr = epoch_count
    summary.learning_cases = len(training_reviews)
    summary.test_cases = len(test_reviews)
    summary.total_success_predictions = verify.count(1)
    summary.total_failed_predictions = verify.count(-1)
    summary.success_rate = verify.count(1) / len(verify)
    summary.fail_rate = verify.count(-1) / len(verify)
    summary.confusion_matrix = confusion_matrix
    summary.training_time = total_time_sec
    network.summary = summary

    log.info('--------------------------------------------------------')
    log.info(f'Testing finished, epoch {summary.epoch_nr} \n')
    log.info('SUMARY:')
    log.info(f'Total learning cases: {summary.learning_cases}')
    log.info(f'Total testing cases: {summary.test_cases}')
    log.info(f'Total testing scores: {len(verify)}')
    log.info(f'Total success predictions: {summary.total_success_predictions}')
    log.info(f'Total failed predictions: {summary.total_failed_predictions}')
    log.info(f'Success rate: {summary.success_rate}')
    log.info(f'Fail rate: {summary.fail_rate}')
    log.info(f'Confusion matrix: {summary.confusion_matrix}')
    log.info(f'Training time: {summary.training_time}')
    log.info('-------------------------------------------------------')


def save_network_params(network):
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = f'networks/network_imdb_{current_time}.json'
    network_json = jsonpickle.dumps(network.__dict__)

    with open(path, 'w') as file:
        file.write(network_json)

    log.info(f'Network params saved to JSON')


epoch_count = 1
for i in range(0, epoch_count):
    learn_epoch(i + 1)

save_network_params(network)




# INFO:root:Testing finished, epoch 1
# INFO:root:SUMARY:
# INFO:root:Total learnig cases: 24000
# INFO:root:Total testing cases: 1000
# INFO:root:Total testing scores: 1000
# INFO:root:Total success predictions: 829
# INFO:root:Total failed predictions: 171
# INFO:root:Total uncertain predictions: 0
# INFO:root:Success rate: 0.829
# INFO:root:Fail rate: 0.171
#
# INFO:root:Testing finished, epoch 2
# INFO:root:SUMARY:
# INFO:root:Total learnig cases: 24000
# INFO:root:Total testing cases: 1000
# INFO:root:Total testing scores: 1000
# INFO:root:Total success predictions: 827
# INFO:root:Total failed predictions: 173
# INFO:root:Total uncertain predictions: 0
# INFO:root:Success rate: 0.827
# INFO:root:Fail rate: 0.173
# INFO:root:-------------------------------------------------------
#
# INFO:root:--------------------------------------------------------
# INFO:root:Testing finished, epoch 3
# INFO:root:SUMARY:
# INFO:root:Total learnig cases: 24000
# INFO:root:Total testing cases: 1000
# INFO:root:Total testing scores: 1000
# INFO:root:Total success predictions: 827
# INFO:root:Total failed predictions: 173
# INFO:root:Total uncertain predictions: 0
# INFO:root:Success rate: 0.827
# INFO:root:Fail rate: 0.173
# INFO:root:-------------------------------------------------------
#
