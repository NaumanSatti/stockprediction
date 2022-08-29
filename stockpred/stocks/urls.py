from django.urls import path
from . import views
urlpatterns = [
    path('', views.index, name='index'),
    
    path('allcompanies/', views.all_companies, name='all'),
    path('cpsSpec/', views.CPS_spec, name='cpsspec'),

    path('cpscombine/', views.cpscombine, name='cpscombine'),

    path('specmain/', views.specmain, name='specmain')


]
