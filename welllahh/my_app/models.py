from uuid import uuid4

from django.contrib.auth.models import User
from django.db import models

# Create your models here.


class CustomUser(models.Model):
    customer_id = models.UUIDField(primary_key=True, default=uuid4(), editable=False)
    status_choices = [("any", "anonymous"), ("knw", "known")]
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    age = models.IntegerField(blank=True, null=True)
    parent = models.ForeignKey("self", on_delete=models.CASCADE, blank=True, null=True)
    secret_phrase = models.CharField(max_length=100)
    status = models.CharField(max_length=len("anonymous"), choices=status_choices)

    def __str__(self):
        return f"{self.user.username}-{self.status}"


class UserBodyInfo(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid4(), editable=False)
    weight = models.IntegerField()
    muscle_mass = models.IntegerField(blank=True, null=True)
    height = models.IntegerField()
    fat_mass = models.IntegerField(blank=True, null=True)
    check_time = models.DateField()
    custom_user = models.ForeignKey(CustomUser, on_delete=models.CASCADE)


class Token(models.Model):
    token_id = models.UUIDField(primary_key=True, default=uuid4(), editable=False)
    token = models.CharField(max_length=255)
    custom_user = models.ForeignKey(CustomUser, on_delete=models.CASCADE)


class NutritionType(models.TextChoices):
    CALORY = "Calory", "Calory"
    BLOOD_SUGAR = "Blood Sugar", "Blood Sugar"
    PROTEIN = "Protein", "Protein"
    CHOLESTEROL = "CHOLESTEROL", "CHOLESTEROL"
    URIC_ACID = "URIC_ACID", "URIC_ACID"
    BLOOD_PRESSURE = "BLOOD_PRESSURE", "BLOOD_PRESSURE"


class NutritionProgress(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid4(), editable=False)
    nutrition_type = models.CharField(
        max_length=20, choices=NutritionType.choices, default=NutritionType.CALORY
    )
    check_time = models.DateTimeField()
    value = models.DecimalField(max_digits=10, decimal_places=2)
    user = models.ForeignKey(CustomUser, on_delete=models.CASCADE)

    def str(self):
        return f"{self.nutrition_type} - {self.value} at {self.check_time}"
