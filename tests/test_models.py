from django.test import TestCase
from django.contrib.auth.models import User
from core.models import SkinAnalysis, Product, AnalysisResult

class ModelTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username='testuser', password='password')
        self.product = Product.objects.create(
            name="Test Cream",
            price="100",
            product_url="http://test.com",
            skin_concern="Acne"
        )

    def test_product_str(self):
        self.assertEqual(str(self.product), "Test Cream")

    def test_skin_analysis_creation(self):
        analysis = SkinAnalysis.objects.create(user=self.user, request_type="mobile")
        self.assertEqual(analysis.user.username, "testuser")
        self.assertIn("Scan", str(analysis))
