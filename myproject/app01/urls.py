from django.urls import path
from . import views
urlpatterns = [
  path('register/',views.register,name='register'),
  path('',views.login,name='login'),
  path('table',views.table_index,name='table'),
]