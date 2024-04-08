import math

import numpy
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

print(f'Reviews nr: {reviews_count}')

reviews_scores = []

for i in range(reviews_count):
    reviews_scores.append((reviews[i], scores[i]))

assert len(reviews_scores) == reviews_count

# create tokens set

vocabulary_raw = set()
vocabulary = set()

for review in reviews:
    vocabulary_raw.update(review.split(" "))

print(f'Created raw set of {len(vocabulary_raw)}')

for item in vocabulary_raw:
    if len(item.strip()) > 1:
        vocabulary.add(item.strip())

print(f'Created cleaned set of {len(vocabulary)}')

vocabulary_list = list(vocabulary)

vocabulary_size = len(vocabulary_list)

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

print(review_vectors[0])

assert len(review_vectors) == reviews_count


def map_score(label):
    if label.strip() == 'positive':
        return [1]
    if label.strip() == 'negative':
        return [0]


print('Mapped scores')

# create scores vector
scores = list(map(lambda p: map_score(p), scores))

assert len(review_vectors) == len(scores)

# select training data

# training_reviews = review_vectors[0:23999]
# training_scores = scores[0:23999]

training_reviews = review_vectors[0:23000]
training_scores = scores[0:23000]

print('Created training data')

# select test data
test_reviews = review_vectors[23000:25000]
test_scores = scores[23000:25000]

print('Created test data')

# create network

layers = [vocabulary_size, 4, 1]
dropout = [0, 0]
network = Network(layers, dropout, False)

print(f'test positives: {test_scores.count([1])}')
print(f'test negatives: {test_scores.count([0])}')

print('Network ready')

print(f'Train reviews: {len(training_reviews)}')
print(f'Train scores: {len(training_scores)}')

iterations = len(training_reviews)
iter_nr = 0

for i in range(iterations):
    network.learn(training_reviews[i], training_scores[i])
    print(f'Training iteration {iter_nr} of {iterations}')
    iter_nr +=1

print('Learning finished')


verify = []

for i in range(len(test_reviews)):
    predicted = network.predict(test_reviews[i], False)
    real = test_scores[i]
    print(f'Prediction: {predicted}')
    print(f'Actual: {test_scores[i]}')

    if (real == [1] and predicted[0] > 0.5) or (real == [0] and predicted[0] < -0.5):
        verify.append(1)
    elif (predicted[0] <= 0.5) and (predicted[0] >= -0.5):
        verify.append(0)
    else:
        verify.append(-1)

print('Testing finished')
print('--------------------------------------------------------')
print('SUMARY:')
print(f'Total learnig cases: {len(training_reviews)}')
print(f'Total testing cases: {len(test_reviews)}')
print(f'Total testing scores: {len(verify)}')
print(f'Total success predictions: {verify.count(1)}')
print(f'Total failed predictions: {verify.count(-1)}')
print(f'Total uncertain predictions: {verify.count(0)}')
print(f'Success rate: {verify.count(1)/len(verify)}')
print(f'Fail rate: {verify.count(-1)/len(verify)}')
print(f'Uncertain rate: {verify.count(0)/len(verify)}')








