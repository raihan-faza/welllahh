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
    path('', views.landing_page, name="home"),
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
    path("health_check/nutrition", views.get_health_condition_based_on_nutrition_val, name="get_health_condition_based_on_nutrition_val"),
    path("health_progress/nutrition", views.catat_nutrisi, name="catat_nutrisi"),
    path("health_progress/bmi", views.catat_tinggi_berat, name="catat_tinggi_berat")
]


if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
