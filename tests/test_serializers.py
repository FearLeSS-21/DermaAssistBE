from django.test import TestCase
from core.models import Product
from core.serializers import ProductSerializer

class SerializerTests(TestCase):
    def setUp(self):
        self.product = Product.objects.create(
            name="Serum", price="200", product_url="http://x.com", skin_concern="Wrinkles"
        )

    def test_product_serializer(self):
        serializer = ProductSerializer(self.product)
        data = serializer.data
        self.assertEqual(data['name'], "Serum")
        self.assertEqual(data['skin_concern'], "Wrinkles")
