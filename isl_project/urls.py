from django.contrib import admin
from django.urls    import path, include
from isl_api.views import index, call_page, manifest
 
urlpatterns = [
    path('',       index),
    path('admin/', admin.site.urls),
    path('api/',   include('isl_api.urls')),
    path('call/', call_page),
    path('manifest.json', manifest),
]
