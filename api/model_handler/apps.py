from django.apps import AppConfig
from django.conf import settings
import os
import pickle

from modules.model import ModelHandler
from modules.dist_map import DistributionMap


class ModelHandlerConfig(AppConfig):
    name = 'model_handler'
    model_path = './model_handler/models/model.pkl'
    vectorizer_path = './model_handler/models/vectorizer.pkl'
    ratinger_path = './model_handler/models/rating.pkl'
    model = ModelHandler(model_path=model_path,
    					 vectorizer_path=vectorizer_path,
    					 ratinger_path=ratinger_path) 