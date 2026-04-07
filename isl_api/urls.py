from django.urls import path
from . import views
 
urlpatterns = [
    path('health/',           views.health_check,     name='health'),
    path('gestures/',         views.list_gestures,    name='gestures'),
    path('model-info/',       views.model_info,       name='model-info'),
    path('predict-sequence/', views.predict_sequence, name='predict-sequence'),
]
 
