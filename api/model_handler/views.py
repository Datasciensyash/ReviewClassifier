from django.shortcuts import render
from .apps import ModelHandlerConfig
from django.http import JsonResponse
from rest_framework.views import APIView

class call_model(APIView):
    def get(self, request):
        if request.method == 'GET':
            #Get input data
            input_data = request.GET.get('input')

            print(f'Input data {input_data}')

            print('Trying to get predictions...')

            #Get predictions
            predictions = ModelHandlerConfig.model(input_data)

            print(f'Predictions: {predictions} ')

            #Build response
            predictions_list = []
            for prediction in predictions:
            	predictions_list.append({
            		'Class': int(prediction[0]),
            		'Description': {0:'Neutral', 1:'Positive', -1:'Negative'}[int(prediction[0])],
            		'Rating': round(prediction[1], 1),
            		'Rating_rounded': int(round(prediction[1]))
            		})
            response = {'Predictions': predictions_list}

            print(f'Response: {response}')

            return JsonResponse(response)