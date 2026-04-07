# Add this to your existing isl_api/views.py

def call_page(request):
    """Serves the WebRTC video call page."""
    import os
    from django.http import HttpResponse
    html_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        'templates', 'call.html'
    )
    with open(html_path, 'r', encoding='utf-8') as f:
        return HttpResponse(f.read(), content_type='text/html')


# Add this URL to isl_project/urls.py:
# from isl_api.views import index, call_page
#
# urlpatterns = [
#     path('',      index),
#     path('call/', call_page),    ← ADD THIS
#     path('admin/', admin.site.urls),
#     path('api/',   include('isl_api.urls')),
# ]
