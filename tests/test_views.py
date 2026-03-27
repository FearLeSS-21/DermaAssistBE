from rest_framework.test import APITestCase
from django.urls import reverse
from core.models import Product

class ViewTests(APITestCase):
    def setUp(self):
        Product.objects.create(name="Alpha Cream", price="50", product_url="#", skin_concern="Acne")
        Product.objects.create(name="Beta Gel", price="60", product_url="#", skin_concern="Wrinkles")

    def test_product_search(self):
        url = reverse('product_search')
        # APIClient returns data as dict, not bytes
        response = self.client.get(url, {'q': 'Alpha'})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.data), 1)
        self.assertEqual(response.data[0]['name'], "Alpha Cream")

    def test_product_search_empty(self):
        url = reverse('product_search')
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)
        self.assertTrue(len(response.data) > 0)
