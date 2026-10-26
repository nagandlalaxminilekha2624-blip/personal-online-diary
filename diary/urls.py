from django.contrib.auth import views as auth_views
from django.urls import path

from . import views

urlpatterns = [
    # accounts
    path("accounts/login/", auth_views.LoginView.as_view(), name="login"),
    path("accounts/logout/", auth_views.LogoutView.as_view(), name="logout"),
    path("accounts/signup/", views.signup, name="signup"),
    # diary
    path("", views.entry_list, name="entry_list"),
    path("dashboard/", views.dashboard, name="dashboard"),
    path("entry/new/", views.entry_create, name="entry_create"),
    path("entry/<int:pk>/", views.entry_detail, name="entry_detail"),
    path("entry/<int:pk>/edit/", views.entry_edit, name="entry_edit"),
    path("entry/<int:pk>/delete/", views.entry_delete, name="entry_delete"),
    # machine learning
    path("predict/", views.predict_mood, name="predict_mood"),
]
