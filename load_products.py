import os 
import django 
import csv 
import sys 

os.environ.setdefault('DJANGO_SETTINGS_MODULE','config.settings')
django.setup()

from core.models import Product 

CSV_PATH =os.path.join('web_scraping','eparkville_skincare_playwright.csv')

def run():
    if not os.path.exists(CSV_PATH ):
        print(f"Error: CSV file not found at {CSV_PATH}")
        return 

    print("Cleaning old products...")
    Product.objects.all().delete()

    print(f"Loading products from {CSV_PATH}...")
    with open(CSV_PATH,'r', encoding ='utf-8') as f :
        reader =csv.DictReader(f )
        count =0 
        for row in reader :
            try :
                Product.objects.create(
                name =row['name'].replace('-',' ').title(),
                price =row['price'],
                product_url =row['url'],
                skin_concern =row['target'],
                )
                count +=1 
            except Exception as e :
                print(f"Skipping row: {row}.Error: {e}")

    print(f"Successfully loaded {count} products!")

if __name__ =='__main__':
    run()
