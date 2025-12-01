from django.urls import path
from .views import (
    UploadImageView, ProgressView, UserProfileView,
    AllReports, SignupView, LoginView, LogoutView,
    ProductSearchView, ProgressCompare, Results
)

urlpatterns = [
    path('upload/', UploadImageView.as_view(), name='upload'),
    path('progress/', ProgressView.as_view(), name='progress'),
    path('profile/', UserProfileView.as_view(), name='profile'),
    path('reports/', AllReports.as_view(), name='reports'),
    path('signup/', SignupView.as_view(), name='signup'),
    path('login/', LoginView.as_view(), name='login'),
    path('logout/', LogoutView.as_view(), name='logout'),
    path('search/', ProductSearchView.as_view(), name='search_products'),
    path('progresscompare/', ProgressCompare.as_view(), name='progresscompare'),
    path('results/', Results.as_view(), name='results'),
]