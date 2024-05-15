"""
URL configuration for myproject project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.0/topics/http/urls/
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
from django.contrib import admin
from django.urls import path,include
from myproject import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('home', views.home, name='home'),
    path('sample_10000',views.sample_10000,name='sample_10000'),
    path('sample_10000_visibility_severity',views.sample_10000_visibility_severity,name='sample_10000_visibility_severity'),
    path('sample_10000_distance_severity',views.sample_10000_distance_severity,name='sample_10000_distance_severity'),
    # path('test_table',views.get_main_item,name='test_table'),
    path('user_information',views.user_information,name='user_information'),
    path('',include('app01.urls')),
    path('severity_predict',views.severity_predict,name='severity_predict'),
    path('result', views.result, name='result'),
    path('csv_severity_predict',views.csv_severity_predict,name='csv_severity_predict'),
    path('result2',views.result2,name='result2'),
    path('fields_severity_predict',views.fields_severity_predict,name='fields_severity_predict'),
    path('accident_sum_state',views.sum_state,name='accident_sum_state'),
    #聚类
    path('clustering_location',views.clustering_location,name='clustering_location'),
    path('state_severity_kmeans',views.state_kmeans,name='state_severity_kmeans'),
    path('sample_1000_',views.sample_1000,name='sample_1000_'),
    path('sample_1000_kmeans',views.sample_1000_kmeans,name='sample_1000_kmeans'),
    path('sample_1000_hierarchial',views.sample_1000_hierarchial,name='sample_1000_hierarchial'),
    path('clustering_time',views.clustering_time,name='clustering_time'),
    #时序
    path('time_series_city',views.time_series_city,name='time_series_city'),
    path('important_city_location',views.important_city_location,name='important_city_location'),
    #path 第一个与url中的名称一致，第二个是view的函数
    path('city_model_change',views.city_model_change,name='city_model_change'),
]
