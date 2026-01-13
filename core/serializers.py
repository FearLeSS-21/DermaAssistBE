from rest_framework import serializers
from django.contrib.auth.models import User
from .models import SkinAnalysis, AnalysisResult, Product

class UserSerializer(serializers.ModelSerializer):
    password = serializers.CharField(write_only=True)

    class Meta:
        model = User
        fields = ('id', 'username', 'email', 'password')

    def create(self, validated_data):
        user = User.objects.create_user(
            username=validated_data['username'],
            email=validated_data.get('email', ''),
            password=validated_data['password']
        )
        return user

class AnalysisResultSerializer(serializers.ModelSerializer):
    class Meta:
        model = AnalysisResult
        fields = ['timestamp', 'acne_count', 'wrinkle_score', 'eyebag_score', 'acne_data', 'wrinkles_data']

class SkinAnalysisSerializer(serializers.ModelSerializer):
    result = AnalysisResultSerializer(read_only=True)
    image_url = serializers.SerializerMethodField()

    class Meta:
        model = SkinAnalysis
        fields = ['id', 'created_at', 'image_url', 'result']

    def get_image_url(self, obj):
        request = self.context.get('request')
        if obj.image and request:
            return request.build_absolute_uri(obj.image.url)
        return None

class ProductSerializer(serializers.ModelSerializer):
    class Meta:
        model = Product
        fields = ['id', 'name', 'price', 'product_url', 'skin_concern', 'image_url']