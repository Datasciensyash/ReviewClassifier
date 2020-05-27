from typing import Callable, Union, Tuple
import numpy as np 
import pickle

class ModelHandler():
	"""
	Оборачивает модель удобным API.

	"""

	def __init__(self, model_path:str, vectorizer_path:str) -> None:
		"""
		:model_path: (str) Путь к модели, должна быть в формате .pkl 
		:vectorizer_path: (str) Путь к векторизатору, должен быть в формате .pkl
		"""

		self.model = self.load_pickle(model_path)
		self.vectorizer = self.load_pickle(vectorizer_path)

	def load_pickle(self, path:str) -> Callable:
		"""
		Загружает файл в формате .pkl и возвращает его.
		:path: (str) -> Путь к файлу
		"""
		with open(path, 'rb') as pickled_file:
			return pickle.load(pickled_file)

	def predict(self, data:str) -> Tuple[np.array]:
		"""
		Выдает предсказания, используя векторизатор и модель.
		:data: (str) -> Данные для предсказания.
			Формат: "This is a good film!"

		Возвращает: tuple(class:np.array, rating:np.array)
		"""

		#Преобразование в необходимый формат
		data = np.array([data])

		#Векторизация
		features = self.vectorizer.transform(data)

		#Получение предсказания модели
		positive_proba = self.model.predict_proba(features)[:, 1]

		#Получение оценки от 1 до 10:
		rating = positive_proba * 10

		#Преобразование вероятности в класс
		review_class = np.round(positive_proba)

		return review_class, rating

	def __call__(self, data:Union[str, list]) -> list:
		"""
		Функция для использования класса как Callable.
		Служит оберткой над self.predict.
		:data: (str) -> Данные для предсказания.
			Формат: "This is a good film!"

		Возвращает: predictions:list
		"""
		review_class, rating = self.predict(data)

		predictions = np.stack([review_class, rating], axis=1).tolist()

		return predictions



