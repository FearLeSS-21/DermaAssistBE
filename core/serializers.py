from rest_framework import serializers 
from django.contrib.auth.models import User 
from.models import SkinAnalysis, AnalysisResult, Product 

<<<<<<< HEAD
class UserSerializer(serializers.ModelSerializer ):
    password =serializers.CharField(write_only =True )
=======
class UserSerializer(serializers.ModelSerializer):
    """
    Serializer for the User model, handling registration and creation.
    """
    password = serializers.CharField(write_only=True)
>>>>>>> 3afbbd23597a33ba8104ff11c6fa0978c4803485

    class Meta :
        model =User 
        fields =('id','username','email','password')

    def create(self, validated_data ):
        user =User.objects.create_user(
        username =validated_data['username'],
        email =validated_data.get('email',''),
        password =validated_data['password']
        )
        return user 

<<<<<<< HEAD
class AnalysisResultSerializer(serializers.ModelSerializer ):
    class Meta :
        model =AnalysisResult 
        fields =['timestamp','acne_count','wrinkle_score','eyebag_score','acne_data','wrinkles_data']

class SkinAnalysisSerializer(serializers.ModelSerializer ):
    result =AnalysisResultSerializer(read_only =True )
    image_url =serializers.SerializerMethodField()
=======
class AnalysisResultSerializer(serializers.ModelSerializer):
    """
    Serializer for AnalysisResult to expose analysis scores and data.
    """
    class Meta:
        model = AnalysisResult
        fields = ['timestamp', 'acne_count', 'wrinkle_score', 'eyebag_score', 'acne_data', 'wrinkles_data']

class SkinAnalysisSerializer(serializers.ModelSerializer):
    """
    Serializer for SkinAnalysis, including related results and image URL.
    """
    result = AnalysisResultSerializer(read_only=True)
    image_url = serializers.SerializerMethodField()
>>>>>>> 3afbbd23597a33ba8104ff11c6fa0978c4803485

    class Meta :
        model =SkinAnalysis 
        fields =['id','created_at','image_url','result']

    def get_image_url(self, obj ):
        request =self.context.get('request')
        if obj.image and request :
            return request.build_absolute_uri(obj.image.url )
        return None 

<<<<<<< HEAD
class ProductSerializer(serializers.ModelSerializer ):
    class Meta :
        model =Product 
        fields =['id','name','price','product_url','skin_concern','image_url']
=======
class ProductSerializer(serializers.ModelSerializer):
    """
    Serializer for Product model.
    """
    class Meta:
        model = Product
        fields = ['id', 'name', 'price', 'product_url', 'skin_concern', 'image_url']
>>>>>>> 3afbbd23597a33ba8104ff11c6fa0978c4803485
