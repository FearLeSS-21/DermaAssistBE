from django.db import models
from django.utils import timezone
import json

class Image(models.Model):

    """Model to store uploaded and processed images."""
    name = models.CharField(max_length=255)
    image = models.ImageField(upload_to="Project_Folder/original/")
    processed_image = models.ImageField(upload_to="Project_Folder/processed/", null=True, blank=True)
    request_type = models.CharField(max_length=10, default="unknown")
    ip_address = models.GenericIPAddressField(null=True, blank=True)
    timestamp = models.DateTimeField(default=timezone.now)

    def __str__(self):
        return f"{self.request_type} for {self.name} at {self.timestamp}"


class UserProfile(models.Model):
    """
    Model to store user profile information.
    The password field is nullable to support password-less accounts if needed.
    """
    user_id = models.CharField(max_length=255, unique=True)
    name = models.CharField(max_length=255, null=True, blank=True)
    password = models.CharField(max_length=255, null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.user_id


class Progress(models.Model):
    """
    Model to store user progress data for skin condition detections.
    """
    user_id = models.CharField(max_length=100)
    timestamp = models.DateTimeField(auto_now_add=True)
    acne = models.JSONField(default=list)
    wrinkles = models.JSONField(default=list)
    eyebags = models.JSONField(default=list)
    eczema = models.JSONField(default=list)
    progress_plot = models.CharField(max_length=255, null=True, blank=True)  # Added to store progress plot path

    def __str__(self):
        return f"Progress for {self.user_id} at {self.timestamp}"


class ProgressJson(models.Model):
    """
    Model to store user progress data in JSON format for skin condition detections.
    """
    user_id = models.CharField(max_length=100)
    timestamp = models.DateTimeField(default=timezone.now)
    data = models.JSONField(default=dict)  # Stores JSON data with detections and image_path

    def __str__(self):
        return f"JSON Progress for {self.user_id} at {self.timestamp}"