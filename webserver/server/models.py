from django.db import models
from django.contrib.auth import get_user_model

# model name should be singular! 
class ContextRecord(models.Model):
    # required
    interest = models.CharField(
            max_length=128,
    )

    # required
    time = models.DateTimeField()

    # get_user_model() returns User or custom user model
    # at import time.
    user = models.ForeignKey(
            get_user_model(), 
            on_delete=models.CASCADE
    )
