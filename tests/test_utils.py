from django.test import TestCase
from core.models import Product
from core.views.utils import recommend_products

class UtilsTests(TestCase):
    def setUp(self):
        Product.objects.create(name="Acne Fix", price="10", product_url="#", skin_concern="Acne")
        Product.objects.create(name="Anti Aging", price="20", product_url="#", skin_concern="Wrinkles")
        Product.objects.create(name="Eye Cream", price="30", product_url="#", skin_concern="Eyebags")
        Product.objects.create(name="General Soap", price="5", product_url="#", skin_concern="Other")

    def test_recommend_acne(self):
        recs = recommend_products(acne_severity=10, wrinkle_severity=0, eyebag_severity=0)
        names =[r['name'] for r in recs]
        self.assertIn("Acne Fix", names)
        self.assertNotIn("Anti Aging", names)

    def test_recommend_mixed(self):
        recs = recommend_products(acne_severity=5, wrinkle_severity=5, eyebag_severity=0)
        names =[r['name'] for r in recs]
        self.assertIn("Acne Fix", names)
        self.assertIn("Anti Aging", names)
