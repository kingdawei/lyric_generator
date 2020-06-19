from django.db import models
from django.contrib.auth.models import User

# Create your models here.


class GeneratedLyrics(models.Model):
    lyrics = models.TextField()
    seed = models.CharField(max_length=255)
    length = models.IntegerField()
    temperature = models.FloatField()
    author = models.ForeignKey(User, on_delete=models.CASCADE, null=True)
