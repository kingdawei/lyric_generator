from django.db import models

# Create your models here.


class GeneratedLyrics(models.Model):
    lyrics = models.TextField()
    seed = models.CharField(max_length=255)
    length = models.IntegerField()
    temperature = models.FloatField()
