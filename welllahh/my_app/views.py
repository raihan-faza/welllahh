from django.contrib.auth.models import User
from django.http.response import JsonResponse
from django.shortcuts import render

# Create your views here.


def register_user(request):
    username = request.POST.get("username")
    password = request.POST.get("password")
    email = request.POST.get("email")
