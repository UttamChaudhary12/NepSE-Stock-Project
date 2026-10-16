from django.urls import path

from . import views

urlpatterns = [
    path('', views.dashboard, name='dashboard'),
    path('stock/<path:symbol>/', views.stock_detail, name='stock'),  # some debenture symbols contain '/'
    path('about/', views.about, name='about'),
    path('api/signals/', views.api_signals, name='api_signals'),
]
