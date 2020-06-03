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

def load_pickle(filename:str) -> pd.DataFrame:
	return pd.read_pickle(filename)

def load_dataframes(filenames_list:list) -> tuple:
	return [load_pickle(filename) for filename in filenames_list]

def get_sweep_config(init_config:dict) -> dict:
	wandb.init(project="review_classifier", config=init_config)
	return wandb.config

def init_vectorizer(min_df:int, max_df:float) -> TfidfVectorizer:
	return TfidfVectorizer(min_df=min_df, max_df=max_df)

def init_model(solver:str, max_iter:int, random_state:int):
	return LogisticRegression(solver=solver, max_iter=max_iter, random_state=random_state)

def get_corpus(df_list:list, colname:str='Text') -> np.array:
	"""
	Concatenates texts of all dataframes in df_list into corpus.
	Args:
		:df_list: list of pd.DataFrame
		:colname: name of column where text is stored
	"""
	return np.concatenate([df['Text'].values for df in df_list])


train_file = 'dataset/train.pkl'
test_file = 'dataset/test.pkl'
unsup_file = 'dataset/unsup.pkl'

random_state = 1024 #Для воспроизведения
val_ratio = 0.2

if __name__ == '__main__':

	config = get_sweep_config(hparams)

	train_df, unsup_df, test_df = load_dataframes([train_file, unsup_file, test_file])

	#Make train-val split
	train_df, val_df = train_test_split(train_df, test_size=val_ratio, random_state=random_state)

	#Vectorizer init
	if config['vectorizer_fit_unsup']:
		corpus = get_corpus([train_df, unsup_df], colname='Text')
	else:
		corpus = get_corpus([train_df], colname='Text')

	vectorizer = init_vectorizer(min_df=config['min_df'], max_df=config['max_df'])

	#Fit vectorizer with train corpus
	vectorizer = vectorizer.fit(corpus)

	#Get train features from vectorizer
	train_corpus = get_corpus([train_df], colname='Text')
	train_features = vectorizer.transform(train_corpus)

	#Model init
	model = init_model(solver=config['solver'], max_iter=config['max_iter'], random_state=random_state)

	model = model.fit(train_features, train_df['Target'])

	if config['pseudolabel_unsup']:
		unsup_corpus = get_corpus([unsup_df], colname='Text')

		unsup_features = vectorizer.transform(unsup_corpus)

		labels_unsup = model.predict_proba(unsup_features).argmax(1)

		model = init_model(solver=config['solver'], max_iter=config['max_iter'], random_state=random_state)

		features = vectorizer.transform(corpus) #This corpus was created earlier.

		model.fit(features, np.concatenate([train_df['Target'], labels_unsup]))

	predictions = model.predict_proba(vectorizer.transform(val_df['Text']))

	#Metrics calculating
	val_accuracy = accuracy_score(val_df['Target'], predictions.argmax(1))
	val_logloss = log_loss(val_df['Target'], predictions)
	predictions_test = model.predict_proba(vectorizer.transform(test_df['Text']))
	test_accuracy = accuracy_score(test_df['Target'], predictions_test.argmax(1))
	test_logloss = log_loss(test_df['Target'], predictions_test)
	
	wandb.log(
		{'val_accuracy': val_accuracy,
		'val_logloss': val_logloss,
		'test_accuracy': test_accuracy,
		'test_logloss': test_logloss,
		})