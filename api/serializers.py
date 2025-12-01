from rest_framework import serializers
from .models import Image, UserProfile, Progress

class ImageSerializer(serializers.ModelSerializer):
    class Meta:
        model = Image
        fields = ['id', 'image', 'processed_image', 'name', 'request_type', 'ip_address', 'timestamp']

class UserProfileSerializer(serializers.ModelSerializer):
    class Meta:
        model = UserProfile
        fields = ['user_id', 'name', 'password', 'created_at'] # Add other fields as needed

class ProgressSerializer(serializers.ModelSerializer):
    class Meta:
        model = Progress
        fields = ['user_id', 'timestamp', 'acne', 'wrinkles', 'eyebags']