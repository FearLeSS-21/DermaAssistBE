from django.urls import path 
from.views import(
UploadImageView,
RegisterView,
LoginView,
ProgressView,
ProductSearchView,
ResultsView 
)

urlpatterns =[
path('auth/register/', RegisterView.as_view(), name ='register'),
path('auth/login/', LoginView.as_view(), name ='login'),

path('upload/', UploadImageView.as_view(), name ='upload'),
path('results/', ResultsView.as_view(), name ='results'),
path('progress/', ProgressView.as_view(), name ='progress'),
path('products/search/', ProductSearchView.as_view(), name ='product_search'),
]