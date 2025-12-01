from django.apps import AppConfig


class ApiConfig(AppConfig):
    """
    Configuration for the API Django app.
    """
    default_auto_field = "django.db.models.BigAutoField"
    name = "api"