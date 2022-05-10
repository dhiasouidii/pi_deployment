from django.db import models


# Create your models here.
class FuelResults(models.Model):
    moteur = models.CharField(max_length=30)
    carburant = models.CharField(max_length=30)
    marque = models.CharField(max_length=30)
    cv = models.IntegerField()
    year = models.IntegerField()
    voiture = models.CharField(max_length=30)
    prediction = models.CharField(max_length=30)


class PriceResults(models.Model):
    #category = models.CharField(max_length=30)
    marque = models.CharField(max_length=30)
    transmission = models.CharField(max_length=30)
    carburant = models.CharField(max_length=30)
    annee = models.IntegerField()
    kilometrage = models.IntegerField()
    age = models.IntegerField()
    prediction = models.CharField(max_length=30)


class Reviews(models.Model):
    make = models.CharField(max_length=30)
    model = models.CharField(max_length=30)
    rating = models.FloatField()
    recommended_pct = models.CharField(max_length=30)
    review_title = models.CharField(max_length=30)
    review_date = models.CharField(max_length=30)
    review_user_name = models.CharField(max_length=30)
    review_user_location = models.CharField(max_length=30)
    review_text = models.CharField(max_length=30)
    Scores = models.CharField(max_length=40)


def __str__(self):
    return self.prediction
