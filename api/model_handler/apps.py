from django.apps import AppConfig
from django.conf import settings
import os
import pickle

from model_handler.modules.model import ModelHandler


class ModelHandlerConfig(AppConfig):
    name = 'model_handler'
    model_path = './model_handler/models/model.pkl'
    vectorizer_path = './model_handler/models/vectorizer.pkl'
    model = ModelHandler(model_path=model_path,
    					 vectorizer_path=vectorizer_path)