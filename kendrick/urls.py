from django.urls import path
from . import views

#app_name = 'kendrick'

urlpatterns = [
    path('', views.main_page, name='kendrick-home'),
    path('results/', views.view_results, name='kendrick-results'),
    path('about/', views.about_page, name='kendrick-about'),
    path('analyze/', views.analyze_review, name='kendrick-request_analysis'),
]
