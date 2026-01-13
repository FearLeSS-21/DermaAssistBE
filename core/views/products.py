from rest_framework.views import APIView
from rest_framework.response import Response
import pandas as pd
import os

from ..models import Product
from ..serializers import ProductSerializer
from django.db.models import Q

class ProductSearchView(APIView):
    def get(self, request):
        query = request.query_params.get('q', '').lower()
        
        if query:
            products = Product.objects.filter(name__icontains=query)
        else:
            products = Product.objects.all()[:20]
            
        serializer = ProductSerializer(products, many=True)
        return Response(serializer.data)