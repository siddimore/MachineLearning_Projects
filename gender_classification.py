import random

from nltk import NaiveBayesClassifier
from nltk.classify import accuracy as nltk_accuracy
from nltk.corpus import names

"""
Extract last `N` letters from Word
"""

def feature_extractor(word, N=2):
    last_n_letters = word[-N:]
    return {'feature': last_n_letters.lower()}



def create_training_data():

    male_list = [{name, 'male'} for name in names.words('male.txt')]
    female_list = [{name, 'female'} for name in names.words('female.txt')]

    return (male_list + female_list)

# Seed data using random number generator

random.seed(5)
#shuffle data
data = create_training_data()
random.shuffle(data)

# Create Test Data
input_names = ['Alexander', 'Danielle', 'DAvid', 'Cheryl']

# define number of smaples used for train and test
num_train = int(0.8 * len(data))

for i in range(1,6):
    print ('\n Number of End Letters: ', i)
    features = [(feature_extractor(n, i), gender) for (n,gender) in data]

    # Seperate Data into training and test
    train_data, test_data = features[:num_train],features[num_train:]

    classifier = NaiveBayesClassifier.train(train_data)

    # computer accuracy of NaiveBayesClassifier
    accuracy = round(100 * nltk_accuracy(classifier, test_data), 2)

    print('Accuracy= ' + str(accuracy) + '%')
    # predict outputs
    for name in input_names:
        print(name,  "==>", classifier.classify(feature_extractor(name, i)))
