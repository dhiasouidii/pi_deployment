from django.shortcuts import render
from django.http import JsonResponse
import pandas as pd
import sklearn
from predict.models import FuelResults, Reviews
from predict.models import PriceResults
from sklearn.preprocessing import LabelEncoder
import json
import warnings
import numpy
from statistics import mean
from django.core.paginator import Paginator
# Imports for our project
from selenium import webdriver
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.firefox.options import Options
from bs4 import BeautifulSoup
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
import pymongo
import requests
import urllib.request
import os
import numpy as np
import imutils
import easyocr
import cv2
from selenium.webdriver.common.keys import Keys
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver import ChromeOptions
from time import sleep
from selenium.common.exceptions import NoSuchElementException

warnings.filterwarnings("ignore", category=numpy.VisibleDeprecationWarning)


def index(request):
    marques = numpy.load(r"D:\esprit\Semestre 2\Projet DS\marque_classes_price.npy", allow_pickle=True)
    models = numpy.load(r"D:\esprit\Semestre 2\Projet DS\model_classes_price.npy", allow_pickle=True)
    # categories = numpy.load(r"D:\esprit\Semestre 2\Projet DS\category_classes.npy", allow_pickle=True)

    return render(request, 'landing.html',
                  {'marques': marques,
                   'models': models,
                   # 'categories': categories
                   })


def fuel_page(request):
    encode_voiture = numpy.load(r"D:\esprit\Semestre 2\Projet DS\voiture_classes.npy", allow_pickle=True)
    encode_moteur = numpy.load(r"D:\esprit\Semestre 2\Projet DS\moteur_classes.npy", allow_pickle=True)
    encode_marque = numpy.load(r"D:\esprit\Semestre 2\Projet DS\marque_classes.npy", allow_pickle=True)

    return render(request, 'fuel_prediction.html',
                  {'voitures': encode_voiture,
                   'moteurs': encode_moteur,
                   'marques': encode_marque})


def sentiments(request):
    data = pd.read_csv(r"static/dataincsv.csv", low_memory=False)
    json_records = data.reset_index().to_json(orient='records')
    brands = data['make'].unique
    data = json.loads(json_records)
    paginator = Paginator(data, 10)
    page = paginator.get_page(1)
    haha = "haha"

    return render(request, 'sentiment_analysis.html',
                  {
                      'brands': brands,
                      'page': page,
                      'page_count': paginator.count
                  })


def get_models(request):
    if request.POST.get('action') == 'post':
        # Receive data from client
        brand = str(request.POST.get('brand'))
        models = Reviews.objects.filter(make=brand).values_list('model').distinct()

        return JsonResponse({'result': list(models)},
                            safe=False)


def get_models_2(request):
    if request.POST.get('action') == 'post':
        # Receive data from client
        brand = str(request.POST.get('marque'))
        models = PriceResults.objects.filter(marque=brand).values_list('modele').distinct()

        return JsonResponse({'result': list(models)},
                            safe=False)


def get_models_fuel(request):
    if request.POST.get('action') == 'post':
        # Receive data from client
        brand = str(request.POST.get('marque'))
        models = FuelResults.objects.filter(marque=brand).values_list('voiture').distinct()

        return JsonResponse({'result': list(models)},
                            safe=False)


def get_moteurs_fuel(request):
    if request.POST.get('action') == 'post':
        # Receive data from client
        voiture = str(request.POST.get('voiture'))
        moteurs = FuelResults.objects.filter(voiture=voiture).values_list('moteur').distinct()
        return JsonResponse({'result': list(moteurs)},
                            safe=False)


def fill_sentiments_list(request):
    if request.POST.get('action') == 'post':
        # Receive data from client
        brand = str(request.POST.get('brand'))
        model = str(request.POST.get('model'))
        reviews = Reviews.objects.filter(make=brand, model=model).values_list()
        return JsonResponse({'reviews': list(reviews)},
                            safe=False)


def predict_fuel(request):
    if request.POST.get('action') == 'post':
        # Receive data from client
        moteur = str(request.POST.get('moteur')).strip()
        carburant = str(request.POST.get('carburant')).strip()
        marque = str(request.POST.get('marque')).strip()
        cv = int(request.POST.get('cv'))
        year = int(request.POST.get('year'))
        voiture = str(request.POST.get('voiture')).strip()
        # Unpickle model
        model = pd.read_pickle(r"D:\esprit\Semestre 2\Projet DS\conso_model.pickle")

        encode_voiture = numpy.load(r"D:\esprit\Semestre 2\Projet DS\voiture_classes.npy", allow_pickle=True)
        encode_moteur = numpy.load(r"D:\esprit\Semestre 2\Projet DS\moteur_classes.npy", allow_pickle=True)
        encode_marque = numpy.load(r"D:\esprit\Semestre 2\Projet DS\marque_classes.npy", allow_pickle=True)

        voitureencoder = LabelEncoder()
        moteurencoder = LabelEncoder()
        marqueencoder = LabelEncoder()

        voitureencoder.classes_ = encode_voiture
        moteurencoder.classes_ = encode_moteur
        marqueencoder.classes_ = encode_marque

        # Make prediction
        result = model.predict([[moteurencoder.transform([moteur]),
                                 carb(carburant),
                                 marqueencoder.transform([marque]),
                                 cv,
                                 year,
                                 voitureencoder.transform([voiture])]])
        prediction = "{:.2f}".format(result[0])

        FuelResults.objects.create(moteur=moteur, carburant=carburant, marque=marque,
                                   cv=cv, year=year, voiture=voiture, prediction=str(prediction))
        return JsonResponse({'result': str(prediction)},
                            safe=False)


def carb(x):
    if x == "Diesel":
        return 1
    elif x == "Essence":
        return 0
    else:
        return 2


def ref1(x):
    if x == "Manuelle":
        return 1
    elif x == "Automatique":
        return 0


def ref2(x):
    if x == "Essence":
        return 1
    elif x == "Diesel":
        return 0


def cat(x):
    if x == "Budget_Friendly":
        return 0
    elif x == "Medium_Range":
        return 1
    elif x == "high_Range":
        return 2
    else:
        return 3


def price_page(request):
    return render(request, 'landing.html')


def predict_price(request):
    range = request.POST.get('price_range').split(",", 2)
    range = mean([int(range[0]), int(range[1])])
    if request.POST.get('action') == 'post':
        # Receive data from client
        # category = str(request.POST.get('category')).strip()
        marque = str(request.POST.get('marque'))
        modele = str(request.POST.get('model'))
        transmission = str(request.POST.get('transmission'))
        carburant = str(request.POST.get('carburant'))
        annee = int(request.POST.get('year'))
        kilometrage = range
        age = 2022 - annee
        # category = cat(category)
        # Unpickle model
        model = pd.read_pickle(r"D:\esprit\Semestre 2\Projet DS\price_model.pickle")
        scaler = pd.read_pickle(r"D:\esprit\Semestre 2\Projet DS\price_scaler.pickle")

        # encode_category = numpy.load(r"D:\esprit\Semestre 2\Projet DS\category_classes.npy", allow_pickle=True)
        encode_marque = numpy.load(r"D:\esprit\Semestre 2\Projet DS\marque_classes_price.npy", allow_pickle=True)
        encode_model = numpy.load(r"D:\esprit\Semestre 2\Projet DS\model_classes.npy", allow_pickle=True)

        # categoryencoder = LabelEncoder()
        marqueencoder = LabelEncoder()
        modelencoder = LabelEncoder()

        # categoryencoder.classes_ = encode_category
        marqueencoder.classes_ = encode_marque
        modelencoder.classes_ = encode_model

        # scaled = scaler.transform([[annee, kilometrage, age]])
        scaled = scaler.transform([[kilometrage, age]])
        # Make prediction
        result = model.predict([[modelencoder.transform([modele]),
                                 # categoryencoder.transform([category]),
                                 marqueencoder.transform([marque]),
                                 ref1(transmission),
                                 ref2(carburant),
                                 scaled[0][0],
                                 scaled[0][1],
                                 # scaled[0][2]
                                 ]])
        prediction = int(result[0])
        PriceResults.objects.create(marque=marque, transmission=transmission, modele=modele,
                                    carburant=carburant, annee=annee, kilometrage=kilometrage,
                                    prediction=str(prediction))

        return JsonResponse({'result': str(prediction)},
                            safe=False)


def notfoundpage(request, exception):
    return render(request, '404.html')


def dashboard(request):
    return render(request, 'dashboard.html')


def detection(request):
    return render(request, 'detection.html')


def detect_car(request):
    # Read in image, Grayscale and Blur
    image = cv2.imread(r'D:\esprit\car.png')
    # convert BGR to Gray
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bfilter = cv2.bilateralFilter(gray, 11, 17, 17)
    # Edge detection
    # canny algorithm allow us to detect edges
    edged = cv2.Canny(bfilter, 30, 200)
    # Find Countours and apply Mask
    # Find shapes (contours) return a tree and aproximate what the contour looks like
    # and store to a variable "keypoints"
    keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contours = imutils.grab_contours(keypoints)
    # return the top 10 contours
    # sorted contour list
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    loaction = None
    for contour in contours:
        # cv2.approxPolydp allow to apporximate the polygon from our contours
        # 10 : skip the little dents and specify its a straight line
        approx = cv2.approxPolyDP(contour, 10, True)
        # if there is 4 lines (Keypoints) its the location of the name plate that we needed
        if len(approx) == 4:
            location = approx
            break
    # blank mask same shape as the gray image
    mask = np.zeros(gray.shape, np.uint8)
    # draw contour "location"
    new_image = (cv2.drawContours(mask, [location], 0, 255, -1))
    # over lay the mask over the original image
    new_image = cv2.bitwise_and(image, image, mask=mask)
    # finding out every single section where our image isn't black
    # and storing theme to variable x , y ...
    (x, y) = np.where(mask == 255)
    # get the max and the min of
    (minX, minY) = (np.min(x), np.min(y))
    (maxX, maxY) = (np.max(x), np.max(y))
    # adding +1 to get a little bit of buffer
    processed_image = gray[minX:maxX + 1, minY:maxY + 1]
    reader = easyocr.Reader(['en'])
    result = reader.readtext(processed_image)
    incrementation = result[0][-2]
    incrementation = incrementation.replace('[', "")
    nenregistrement = result[1][-2]
    print("numéro d'incrémentation:", incrementation)
    print("numéro d'enregistrement:", nenregistrement)
    info=scraping(nenregistrement, incrementation)




    return JsonResponse({'result': incrementation + ' TUNIS ' + nenregistrement,
                         'info':info},
                        safe=False)


def scraping(enregist, increm):
    IDNUM = 'numSerie'
    IDNUMCAR = 'numCar'
    ENTER = '//*[@id="mat-search-btn-container"]/div/button'
    MAKE = '//*[@id="detail-car"]/div/div[1]/div[1]/div[3]'
    DATE = '//*[@id="detail-car"]/div/div[2]/div[1]/div[3]'
    FUEL = '//*[@id="detail-car"]/div/div[1]/div[2]/div[3]'
    POWER = '//*[@id="detail-car"]/div/div[2]/div[2]/div[3]'
    TYPE = '//*[@id="detail-car"]/div/div[3]/div[1]/div[3]'
    BODY = '//*[@id="detail-car"]/div/div[4]/div[1]/div[3]'
    ENGINE = '//*[@id="detail-car"]/div/div[3]/div[2]/div[3]'
    CYLINDERCAP = '//*[@id="detail-car"]/div/div[4]/div[2]/div[3]'

    WEBSITE_URL = 'https://vidange.tn/'
    # Setting up our Web Driver For Selenium
    # I am going to use firefox web driver geckodriver.exe
    driver_service = Service(r'C:\Program Files\geckodriver.exe')
    options = Options()
    # All the work is done in the background : headless=True
    options.headless = True
    # Passing the parameters to the browser
    browser = webdriver.Firefox(service=driver_service, options=options)

    browser.get(WEBSITE_URL)

    SERIAL = browser.find_element_by_id(IDNUM)
    SERIAL.click()
    SERIAL.send_keys(increm)
    NUMCAR = browser.find_element_by_id(IDNUMCAR)
    NUMCAR.click()
    NUMCAR.send_keys(enregist)
    sleep(5)
    NUMCAR.send_keys(Keys.ENTER)
    NUMCAR.send_keys(Keys.ENTER)
    NUMCAR.send_keys(Keys.ENTER)
    NUMCAR.send_keys(Keys.ENTER)
    NUMCAR.send_keys(Keys.ENTER)
    NUMCAR.send_keys(Keys.ENTER)
    NUMCAR.send_keys(Keys.ENTER)
    NUMCAR.send_keys(Keys.ENTER)
    NUMCAR.send_keys(Keys.ENTER)
    NUMCAR.send_keys(Keys.ENTER)

    sleep(5)
    MAKE = browser.find_element_by_xpath(MAKE).text
    DATE = browser.find_element_by_xpath(DATE).text
    FUEL = browser.find_element_by_xpath(FUEL).text
    POWER = browser.find_element_by_xpath(POWER).text
    TYPE = browser.find_element_by_xpath(TYPE).text
    BODY = browser.find_element_by_xpath(BODY).text
    ENGINE = browser.find_element_by_xpath(ENGINE).text
    CYLINDERCAP = browser.find_element_by_xpath(CYLINDERCAP).text

    return [MAKE,DATE,FUEL,POWER,ENGINE]


def dashboard_2(request):
    return render(request, "dashboard_2.html")


def test(request):
    return render(request, "index.html")
