from sklearn.datasets import fetch_20newsgroups
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import logging

# Map Category
category_map = {'talk.politics.misc': 'Politics', 'rec.autos': 'Autos', 'rec.sport.hockey': 'Hockey', 'sci.electronics': 'Electronics', 'sci.med':'Medicine'}

# Get trainig dataset
training_data = fetch_20newsgroups(subset='train',
    categories=category_map.keys(),shuffle=True, random_state=5)

# Build CountVectorizer
count_vectorizer = CountVectorizer()
train_tc = count_vectorizer.fit_transform(training_data.data)
print("\Dimensions of training data: {0}" .format(train_tc.shape))

# create tf-idf Transformer
tfidf = TfidfTransformer()
train_tfidf = tfidf.fit_transform(train_tc)

input_data = ['You need to be careful with cars when drivingon slippery roads',
    'A lot of devices can be operated wirelessly',
    'Players need to be careful when close to goal post',
    'Political debate helps u understand perspectives of both sides']

# Train a MultinomialNB classifier
classifier = MultinomialNB().fit(train_tfidf, training_data.target)

# Transfor input data using count CountVectorizer
input_tc = count_vectorizer.transform(input_data)

# Transform vectorized data using TfidfTransformer
input_tfidf = tfidf.transform(input_tc)

# Predict output categories
predictions = classifier.predict(input_tfidf)

# Print predicted output for  each of the input data
for sent,category in zip(input_data, predictions):
    print('\n Input:', sent, '\n Predicted Category:', \
        category_map[training_data.target_names[category]])
