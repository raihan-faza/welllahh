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
    path("inference/indofood_image", views.inference_indofood_image, name="home"),
    # path("admin/", admin.site.urls),
    path("register", views.register_user, name="register_known_user"),
    path(
        "register_anonymous",
        views.register_anonymous_user,
        name="register_anonymous_user",
    ),
]


if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
