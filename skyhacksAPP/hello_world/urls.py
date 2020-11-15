from django.urls import path
from django.views.generic import TemplateView

from hello_world import views

urlpatterns = [
    path('', views.hello_world, name='home'),
    path('audio', TemplateView.as_view(template_name="audio.html"), name='audio'),
    path('video', TemplateView.as_view(template_name="video.html"), name='video'),
]
