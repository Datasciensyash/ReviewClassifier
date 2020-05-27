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
			prediction = {
				'Class': int(predictions[0][0]),
				'Description': {0:'Neutral', 1:'Positive', -1:'Negative'}[int(predictions[0][0])],
				'Rating': round(predictions[0][1], 1),
				'Rating_rounded': int(round(predictions[0][1]))
				}

			response = {'Predictions': [prediction]}

			return JsonResponse(response)