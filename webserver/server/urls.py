from django.urls import path

from . import views

# preceding / is unnecessary
urlpatterns = [
    path('', views.hello_world, name='hello_world'),
    path('login/', views.user_login, name='login'),
    path('users/<str:username>/', views.user_detail, name='user_detail'),
    path('users/<str:username>/main/', views.main, name='main')
]
