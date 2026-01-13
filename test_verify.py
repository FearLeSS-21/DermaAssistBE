import os
import django
import json

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

from core.models import Product
from core.serializers import ProductSerializer

def verify():
    print("--- Verifying Products ---")
    count = Product.objects.count()
    print(f"Total Products: {count}")
    
    if count == 0:
        print("FAIL: No products found.")
        return

    p = Product.objects.first()
    print(f"Sample Product: {p.name} - {p.skin_concern}")

    print("\n--- Verifying Serializer ---")
    serializer = ProductSerializer(p)
    print(json.dumps(serializer.data, indent=2))
    print("SUCCESS: Serializer works.")

if __name__ == '__main__':
    verify()
