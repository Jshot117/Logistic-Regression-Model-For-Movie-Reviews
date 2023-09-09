import sys
import random
import math
import os
import heapq
import pickle

def read_vocabulary_from_file(filename, n):
    word_freq = {}
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            words = line.strip().split()
            for word in words:
                word_freq[word] = word_freq.get(word, 0) + 1
    
    most_freq_words = heapq.nlargest(n, word_freq, key=word_freq.get)
    return most_freq_words

def review_to_feature_vector(review, vocabulary_set):
    words = set(review.split())
    feature_vector = [1 if word in words else 0 for word in vocabulary_set]
    return feature_vector



def sigmoid(z):
    return 1 / (1 + math.exp(-z))


def train_logistic_regression(training_data, learning_rate, epochs):
    num_features = len(training_data[0][0])
    weights = [0.0] * num_features
    bias = 0.0

    for epoch in range(epochs):
        random.shuffle(training_data)
        for feature_vector, label in training_data:
            z = sum(w * x for w, x in zip(weights, feature_vector)) + bias
            prediction = sigmoid(z)
            error = label - prediction
            weights = [w + learning_rate * error * x for w, x in zip(weights, feature_vector)]
            bias += learning_rate * error

    return weights, bias

def test_logistic_regression(test_data, weights, bias):
    num_correct = 0
    total = len(test_data)
    probabilities = []

    for feature_vector, true_label in test_data:
        z = sum(w * x for w, x in zip(weights, feature_vector)) + bias
        prediction = sigmoid(z)
        predicted_label = 1 if prediction > 0.5 else 0

        if predicted_label == true_label:
            num_correct += 1

        probabilities.append((1 - prediction, prediction)) 
    accuracy = num_correct / total
    return accuracy, probabilities


def read_data_from_directory(directory):
    data = []
    for sentiment in ['pos', 'neg']:
        folder_path = os.path.join(directory, sentiment)
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
                data.append((text, 1 if sentiment == 'pos' else 0))
    return data



if __name__ == "__main__":
    train_data = read_data_from_directory('movie-review-HW2/aclImdb/train')
    test_data = read_data_from_directory('movie-review-HW2/aclImdb/test')
    small_train_data = read_data_from_directory('small_train')
    small_test_data = read_data_from_directory('small_test')
    
    learning_rate = 0.1
    epochs = 1
    vocabulary = read_vocabulary_from_file('movie-review-HW2/aclImdb/imdb.vocab',100)
    set_vocabulary = set(vocabulary)
    small_vocabulary = read_vocabulary_from_file('small_vocabularly',100)
    set_small_vocabulary = set(small_vocabulary)
    # Read training and test data
    train_reviews, train_labels = zip(*train_data)
    test_reviews, test_labels = zip(*test_data)
    small_train_reviews, small_train_labels = zip(*small_train_data)
    small_test_reviews, small_test_labels = zip(*small_test_data)

    # Convert reviews to feature vectors
    train_feature_vectors = [review_to_feature_vector(review, set_vocabulary) for review in train_reviews]
    test_feature_vectors = [review_to_feature_vector(review, set_vocabulary) for review in test_reviews]
    small_train_feature_vectors = [review_to_feature_vector(review, set_small_vocabulary) for review in small_train_reviews]
    small_test_feature_vectors = [review_to_feature_vector(review, set_small_vocabulary) for review in small_test_reviews]

    # Combine feature vectors and labels
    train_data = list(zip(train_feature_vectors, train_labels))
    test_data = list(zip(test_feature_vectors, test_labels))
    small_train_data = list(zip(small_train_feature_vectors, small_train_labels))
    small_test_data = list(zip(small_test_feature_vectors, small_test_labels))
    

    # Train logistic regression
    weights, bias = train_logistic_regression(train_data, learning_rate, epochs)
    small_weights, small_bias = train_logistic_regression(small_train_data, learning_rate, epochs)

    with open('movie-review-BOW.LR', 'wb') as file:
        pickle.dump((weights, bias), file)

    # Test logistic regression
    accuracy, probabilities = test_logistic_regression(test_data, weights, bias)
    small_accuracy, small_probabilities = test_logistic_regression(small_test_data, small_weights, small_bias)
    print("Accuracy:", accuracy)
    print("Probabilites:", probabilities)
    print("Small Accuracy:", small_accuracy)
    print("Small Probabilities:", small_probabilities)
    