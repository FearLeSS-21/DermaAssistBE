from django.db import models 
from django.contrib.auth.models import User 

<<<<<<< HEAD
class SkinAnalysis(models.Model ):
    user =models.ForeignKey(User, on_delete =models.CASCADE, related_name ='scans')
    image =models.ImageField(upload_to ="skin_scans/original/%Y/%m/")
    processed_image =models.ImageField(upload_to ="skin_scans/processed/%Y/%m/", null =True, blank =True )
    created_at =models.DateTimeField(auto_now_add =True )
    request_type =models.CharField(max_length =50, default ="unknown")
    ip_address =models.GenericIPAddressField(null =True, blank =True )

    def __str__(self ):
        return f"Scan {self.id} - {self.user.username}"

class AnalysisResult(models.Model ):
    analysis =models.OneToOneField(SkinAnalysis, on_delete =models.CASCADE, related_name ='result')
    user =models.ForeignKey(User, on_delete =models.CASCADE )
    timestamp =models.DateTimeField(auto_now_add =True )
=======
class SkinAnalysis(models.Model):
    """
    Model representing a skin analysis session initiated by a user.
    """
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='scans')
    image = models.ImageField(upload_to="skin_scans/original/%Y/%m/")
    processed_image = models.ImageField(upload_to="skin_scans/processed/%Y/%m/", null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    request_type = models.CharField(max_length=50, default="unknown")
    ip_address = models.GenericIPAddressField(null=True, blank=True)
    
    def __str__(self):
        return f"Scan {self.id} - {self.user.username}"

class AnalysisResult(models.Model):
    """
    Model storing the quantitative results of a skin analysis.
    """
    analysis = models.OneToOneField(SkinAnalysis, on_delete=models.CASCADE, related_name='result')
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    timestamp = models.DateTimeField(auto_now_add=True)
    
    # Store detailed analysis data as JSON
    acne_data = models.JSONField(default=list)
    wrinkles_data = models.JSONField(default=list)
    eyebags_data = models.JSONField(default=list)
    eczema_data = models.JSONField(default=list)
    
    # Aggregate scores
    acne_count = models.IntegerField(default=0)
    wrinkle_score = models.FloatField(default=0.0)
    eyebag_score = models.FloatField(default=0.0)
>>>>>>> 3afbbd23597a33ba8104ff11c6fa0978c4803485

    acne_data =models.JSONField(default =list )
    wrinkles_data =models.JSONField(default =list )
    eyebags_data =models.JSONField(default =list )
    eczema_data =models.JSONField(default =list )

    acne_count =models.IntegerField(default =0 )
    wrinkle_score =models.FloatField(default =0.0 )
    eyebag_score =models.FloatField(default =0.0 )

    def __str__(self ):
        return f"Results for Scan {self.analysis.id}"

<<<<<<< HEAD
class Product(models.Model ):
    name =models.CharField(max_length =255 )
    price =models.CharField(max_length =50 )
    product_url =models.URLField(max_length =500 )
    skin_concern =models.CharField(max_length =100 )
    image_url =models.URLField(max_length =500, blank =True, null =True )
=======
class Product(models.Model):
    """
    Model representing a skincare product recommended to users.
    """
    name = models.CharField(max_length=255)
    price = models.CharField(max_length=50)
    product_url = models.URLField(max_length=500)
    skin_concern = models.CharField(max_length=100)
    image_url = models.URLField(max_length=500, blank=True, null=True)
>>>>>>> 3afbbd23597a33ba8104ff11c6fa0978c4803485

    def __str__(self ):
        return self.name 