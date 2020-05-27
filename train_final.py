import os

import pickle
import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

import wandb

#Best config from hyperparams search.
hparams = {
	'min_df': 1,
	'max_df': 0.9518,
	'max_iter': 106,
	'solver': 'sag',
	'vectorizer_fit_unsup': False,
	'pseudolabel_unsup': True,

}

train_file = 'dataset/train.pkl'
test_file = 'dataset/test.pkl'
unsup_file = 'dataset/unsup.pkl'

random_state = 1024 #Для воспроизведения

if __name__ == '__main__':

	config = hparams

	train_file = 'dataset/train.pkl'
	unsup_file = 'dataset/unsup.pkl'
	test_file = 'dataset/test.pkl'

	train_df = pd.read_pickle(train_file)
	unsup_df = pd.read_pickle(unsup_file)
	test_df = pd.read_pickle(test_file)

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

	
	predictions_test = model.predict_proba(vectorizer.transform(test_df['Text']))

	test_df['Score'] = predictions_test.T[1]
	test_df.to_csv('Predictions_final.csv', index=False)

	with open('./models/model_final.pkl', 'wb') as f:
		pickle.dump(model, f)

	with open('./models/vectorizer_final.pkl', 'wb') as f:
		pickle.dump(vectorizer, f)	