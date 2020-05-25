import os

import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

import wandb

#Интеграция с Weights&Biases
hparams = {
	'min_df': 1,
	'max_df': 0.9,
	'max_iter': 100,
	'solver': 'sag',
	'vectorizer_fit_unsup': True,
	'pseudolabel_unsup': True,

}

train_file = 'dataset/train.pkl'
test_file = 'dataset/test.pkl'
unsup_file = 'dataset/unsup.pkl'

random_state = 1024 #Для воспроизведения
val_ratio = 0.2

if __name__ == '__main__':

	wandb.init(project="review_classifier", config=hparams)

	config = wandb.config #Получаем конфиг sweep'а.

	train_file = 'dataset/train.pkl'
	unsup_file = 'dataset/unsup.pkl'
	test_file = 'dataset/test.pkl'

	train_df = pd.read_pickle(train_file)
	unsup_df = pd.read_pickle(unsup_file)
	test_df = pd.read_pickle(test_file)

	#Make train-val split
	train_df, val_df = train_test_split(train_df, test_size=val_ratio, random_state=random_state)

	#Vectorizer init
	if config['vectorizer_fit_unsup']:
		corpus = np.concatenate([train_df['Text'].values, unsup_df['Text'].values])
	else:
		corpus = train_df['Text'].values

	vectorizer = TfidfVectorizer(min_df=config['min_df'], max_df=config['max_df'])

	#Train features init
	vectorizer = vectorizer.fit(corpus)

	train_features = vectorizer.transform(train_df['Text'])

	#Model init
	model = LogisticRegression(solver=config['solver'], max_iter=config['max_iter'], random_state=random_state)

	model = model.fit(train_features, train_df['Target'])

	if config['pseudolabel_unsup']:
		unsup_features = vectorizer.transform(unsup_df['Text'])
		labels_unsup = model.predict_proba(unsup_features).argmax(1)

		model = LogisticRegression(solver=config['solver'], max_iter=config['max_iter'], random_state=random_state)

		features = vectorizer.transform(np.concatenate([train_df['Text'], unsup_df['Text']]))

		model.fit(features, np.concatenate([train_df['Target'], labels_unsup]))

	predictions = model.predict_proba(vectorizer.transform(val_df['Text']))

	val_accuracy = accuracy_score(val_df['Target'], predictions.argmax(1))
	val_logloss = log_loss(val_df['Target'], predictions)

	#Only for information
	predictions_test = model.predict_proba(vectorizer.transform(test_df['Text']))
	test_accuracy = accuracy_score(test_df['Target'], predictions_test.argmax(1))
	test_logloss = log_loss(test_df['Target'], predictions_test)
	
	wandb.log(
		{'val_accuracy': val_accuracy,
		'val_logloss': val_logloss,
		'test_accuracy': test_accuracy,
		'test_logloss': test_logloss,
		})