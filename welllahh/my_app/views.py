import random
import string
from uuid import uuid4

from django.contrib.auth import authenticate, login
from django.contrib.auth.models import User
from django.db.models import Q
from django.http.response import JsonResponse
from django.shortcuts import redirect, render

from .models import CustomUser

# Create your views here.


def register_user(request):
    if request.method == "POST":
        username = request.POST.get("username")
        password = request.POST.get("password")
        email = request.POST.get("email")

        user_exists = CustomUser.objects.filter(
            Q(username=username) | Q(email=email)
        ).exists()
        if user_exists:
            return JsonResponse(
                {"message": "Username tersebut sudah ada, ganti username lain yaa."}
            )
        user = User(username=username, password=password)
        user.save()
        custom_user = CustomUser(user=user, email=email)
        custom_user.save()
        login(request, user)
        return redirect(request, "index.html")
    return render(request, "register.html")


def register_anonymous_user(request):
    if request.method == "POST":
        passphrase = request.POST.get("passphrase")
        custom_user = CustomUser.objects.filter(passphrase=passphrase)
    return render(request, "anonymous_login.html")


def normal_login(request):
    username = request.POST.get("login")
    password = request.POST.get("password")
    pass


def login_using_passphrase(request):
    pass


def generate_login_token(request):
    pass


def generate_random_string(length=12):
    characters = string.ascii_letters + string.digits
    return "".join(random.choice(characters) for _ in range(length))
