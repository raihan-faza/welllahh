"""
URL configuration for welllahh project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""

from django.conf import settings
from django.conf.urls.static import static
from django.urls import path

from . import views

app_name = "my_app"

urlpatterns = [
    path("", views.landing_page, name="home"),
    path("ai/food_nutrition/", views.inference_indofood_image, name="food_nutrition"),
    path("meal_plan/", views.meal_plan, name="meal_plan"),
    # path("admin/", admin.site.urls),
    path("register", views.register_user, name="register_known_user"),
    path(
        "register_anonymous",
        views.register_anonymous_user,
        name="register_anonymous_user",
    ),
    path("health_check/bmi", views.get_bmi_condition, name="get_bmi_condition"),
    path(
        "health_check/nutrition",
        views.get_health_condition_based_on_nutrition_val,
        name="get_health_condition_based_on_nutrition_val",
    ),
    path("health_progress/nutrition", views.catat_nutrisi, name="catat_nutrisi"),
    path("health_progress/bmi", views.catat_tinggi_berat, name="catat_tinggi_berat"),
    path("login", views.normal_login, name="normal_login"),
    path("anonymous_login", views.login_using_passphrase, name="anonymous_login"),
    path("ai/chatbot", views.get_chatbot_response, name="chatbot"),
    path("ai/chatbotpage", views.chatbot_page, name="chatbotpage"),
    path("ai/chatbotpage/<uuid:id>", views.chatbot_page, name="chatbotpage"),
    path("dashboard", views.dashboard, name="dashboard"),
    path("delete_nutrition/<uuid:pk>", views.delete_nutrition, name="delete_nutrition"),
    path("logout", views.logout_view, name="logout"),
    path("riwayat", views.riwayat_penyakit, name="riwayat"),
    path("add_nutrition", views.add_nutrition, name="add_nutrition"),
    path("add_riwayat", views.add_riwayat, name="add_riwayat"),
    path("delete_riwayat/<int:pk>", views.delete_riwayat, name="delete_riwayat"),
    path("dashboard_cantik", views.dashboard_cantik, name="dashboard_cantik"),
]


if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
