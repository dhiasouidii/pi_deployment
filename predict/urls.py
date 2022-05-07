from django.urls import path

import predict
from predict import views
from django.conf.urls import handler404
app_name = 'predict'

urlpatterns = [

    path('', views.index, name='index'),

    path('fuel', views.fuel_page, name='fuel_index'),
    path('price', views.price_page, name='price_index'),
    path('sentiments', views.sentiments, name='sa_index'),

    path('fuelpredict', views.predict_fuel, name='fuelpredict'),
    path('pricepredict', views.predict_price, name='pricepredict'),
    path('get_selected_model', views.get_models, name='get_selected_model'),
    path('fill_sentiments_list', views.fill_sentiments_list, name='fill_sentiments_list'),

    path('dashboard', views.dashboard, name='dashboard'),
    path('dashboard_2', views.dashboard_2, name='dashboard2'),
    path('detection', views.detection, name='detection'),
    path('detect_car', views.detect_car, name='detect_car')
]

handler404 = 'predict.views.notfoundpage'
