"""cv_api URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/1.11/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  url(r'^$', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  url(r'^$', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.conf.urls import url, include
    2. Add a URL to urlpatterns:  url(r'^blog/', include('blog.urls'))
"""
from django.urls import include, re_path
from django.contrib import admin

import emotion_detector.views
import gan_detector.views
from django.urls import include, path

import liveness_detector.views

urlpatterns = [
    re_path(r'^gan_detection/detect/$', gan_detector.views.detect),
    re_path(r'^emotion_detection/detect/$', emotion_detector.views.detect),
    re_path(r'^liveness_detection/detect/$', liveness_detector.views.detect),
    path(r'^admin/', admin.site.urls),
]