from django.shortcuts import render
from .apps import ModelHandlerConfig
from django.http import JsonResponse
from rest_framework.views import APIView

class call_model(APIView):
    def get(self, request):
        if request.method == 'GET':
            
            #Get input data
            input_data = request.GET.get('input')

            #Get predictions
            predictions = ModelHandlerConfig.model(input_data)

            #Build response
            predictions_list = []
            for prediction in predictions:
            	predictions_list.append({
            		'Class': int(prediction[0]),
            		'Description': 'Positive' if prediction[0] == 1 else 'Negative',
            		'Rating': round(prediction[1], 2),
            		'Rating_rounded': int(round(prediction[1]))
            		})

            response = {'Predictions': predictions_list}

            # Return response
            return JsonResponse(response)