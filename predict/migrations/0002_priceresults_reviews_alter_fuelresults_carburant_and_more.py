# Generated by Django 4.0.4 on 2022-05-05 14:30

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('predict', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='PriceResults',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('category', models.CharField(max_length=30)),
                ('marque', models.CharField(max_length=30)),
                ('transmission', models.CharField(max_length=30)),
                ('carburant', models.CharField(max_length=30)),
                ('annee', models.IntegerField()),
                ('kilometrage', models.IntegerField()),
                ('age', models.IntegerField()),
                ('prediction', models.CharField(max_length=30)),
            ],
        ),
        migrations.CreateModel(
            name='Reviews',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('make', models.CharField(max_length=30)),
                ('model', models.CharField(max_length=30)),
                ('rating', models.FloatField()),
                ('recommended_pct', models.CharField(max_length=30)),
                ('review_title', models.CharField(max_length=30)),
                ('review_date', models.CharField(max_length=30)),
                ('review_user_name', models.CharField(max_length=30)),
                ('review_user_location', models.CharField(max_length=30)),
                ('review_text', models.CharField(max_length=30)),
                ('Scores', models.CharField(max_length=40)),
            ],
        ),
        migrations.AlterField(
            model_name='fuelresults',
            name='carburant',
            field=models.CharField(max_length=30),
        ),
        migrations.AlterField(
            model_name='fuelresults',
            name='cv',
            field=models.IntegerField(),
        ),
        migrations.AlterField(
            model_name='fuelresults',
            name='prediction',
            field=models.CharField(max_length=30),
        ),
        migrations.AlterField(
            model_name='fuelresults',
            name='year',
            field=models.IntegerField(),
        ),
    ]