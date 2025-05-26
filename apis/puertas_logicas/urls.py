from django.urls import path

from .api import (
    HelloWorld, 
    Train, Calculate, 
)

app_name = "puertas_logicas"
urlpatterns = [
    path("hello_world/", HelloWorld.as_view(), name=f"{app_name}_hello_world"),
    path("train/", Train.as_view(), name=f"{app_name}_train"),
    path("calculate/", Calculate.as_view(), name=f"{app_name}_calculate"),
]