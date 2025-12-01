from django.contrib import admin
from .models import Image, UserProfile, Progress


@admin.register(Image)
class ImageAdmin(admin.ModelAdmin):
    """
    Admin configuration for the Image model.
    """
    list_display = ["name", "request_type", "timestamp"]
    search_fields = ["name", "request_type"]


@admin.register(UserProfile)
class UserProfileAdmin(admin.ModelAdmin):
    """
    Admin configuration for the UserProfile model.
    """
    list_display = ["user_id", "name", "created_at"]
    search_fields = ["user_id", "name"]


@admin.register(Progress)
class ProgressAdmin(admin.ModelAdmin):
    """
    Admin configuration for the Progress model.
    """
    list_display = ["user_id", "timestamp"]
    search_fields = ["user_id"]