<<<<<<< HEAD:web_scraping/scraper.py
import sys 
import os 
import requests 
from bs4 import BeautifulSoup 
from selenium import webdriver 
from selenium.webdriver.chrome.service import Service 
from selenium.webdriver.chrome.options import Options 
from selenium.webdriver.common.by import By 
from selenium.webdriver.support.ui import WebDriverWait 
from selenium.webdriver.support import expected_conditions as EC 
from webdriver_manager.chrome import ChromeDriverManager 
import pandas as pd 
import time 
import logging 
import urllib.parse 
=======
import sys
import os
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd
import time
import logging
import urllib.parse
from typing import List, Dict, Optional, Any
>>>>>>> 3afbbd23597a33ba8104ff11c6fa0978c4803485:webscraping/scraper.py

# Configure logging to file and console with UTF-8 encoding
logging.basicConfig(
<<<<<<< HEAD:web_scraping/scraper.py
level =logging.DEBUG,
format ="%(asctime)s - %(levelname)s - %(message)s",
handlers =[
logging.FileHandler("scraper_log.txt", encoding ="utf-8"),
logging.StreamHandler(sys.stdout )
]
)
logger =logging.getLogger()

sys.stdout.reconfigure(encoding ="utf-8")

def scrape_with_requests(url ):
    headers ={
    "User-Agent":"Mozilla/5.0(Windows NT 10.0; Win64; x64) AppleWebKit/537.36(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
=======
    level=logging.DEBUG,  # DEBUG level to see card HTML
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("scraper_log.txt", encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Force UTF-8 encoding for console output
sys.stdout.reconfigure(encoding="utf-8")

def scrape_with_requests(url: str) -> Optional[BeautifulSoup]:
    """
    Scrape a URL using the requests library.

    Args:
        url (str): The URL to scrape.

    Returns:
        Optional[BeautifulSoup]: A BeautifulSoup object if successful, None otherwise.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
>>>>>>> 3afbbd23597a33ba8104ff11c6fa0978c4803485:webscraping/scraper.py
    }
    try :
        response =requests.get(url, headers =headers, timeout =30 )
        response.raise_for_status()
        soup =BeautifulSoup(response.text,"html.parser")
        return soup 
    except requests.RequestException as e :
        logger.error(f"Request failed for {url}: {str(e)}")
        return None 

<<<<<<< HEAD:web_scraping/scraper.py
def scrape_with_selenium(url ):
    chrome_options =Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    driver =webdriver.Chrome(service =Service(ChromeDriverManager().install()), options =chrome_options )

    try :
        driver.get(url )
        logger.info(f"Page loaded with status: {driver.execute_script('return document.readyState')}")

        WebDriverWait(driver, 60 ).until(
        EC.presence_of_element_located((By.CSS_SELECTOR,".product-grid,.products,.grid-uniform"))
        )
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(5 )

        soup =BeautifulSoup(driver.page_source,"html.parser")
        return soup 
    except Exception as e :
        logger.error(f"Selenium error for {url}: {str(e)}")
        return None 
    finally :
        driver.quit()

def extract_products(soup, page_num ):
    all_products =[]
    if not soup :
        return all_products 

    product_cards =soup.select(".product-card-renderer,.grid__item,.product,.card, article,.product-item")
    if not product_cards :
=======
def scrape_with_selenium(url: str) -> Optional[BeautifulSoup]:
    """
    Scrape a URL using Selenium Webdriver (headless Chrome).

    Args:
        url (str): The URL to scrape.

    Returns:
        Optional[BeautifulSoup]: A BeautifulSoup object if successful, None otherwise.
    """
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    
    try:
        service = Service(ChromeDriverManager().install())
        with webdriver.Chrome(service=service, options=chrome_options) as driver:
            driver.get(url)
            logger.info(f"Page loaded with status: {driver.execute_script('return document.readyState')}")

            # Wait for dynamic content
            WebDriverWait(driver, 60).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, ".product-grid, .products, .grid-uniform"))
            )
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(5)  # Pause to allow rendering

            # Get page source after dynamic content loads
            soup = BeautifulSoup(driver.page_source, "html.parser")
            return soup
    except Exception as e:
        logger.error(f"Selenium error for {url}: {str(e)}")
        return None

def extract_products(soup: BeautifulSoup, page_num: int) -> List[Dict[str, str]]:
    """
    Extract product information from a BeautifulSoup object.

    Args:
        soup (BeautifulSoup): The parsed HTML content.
        page_num (int): The current page number being processed.

    Returns:
        List[Dict[str, str]]: A list of dictionaries containing product details (name, price, url).
    """
    all_products = []
    if not soup:
        return all_products

    # Try common Shopify product card structures
    product_cards = soup.select(".product-card-renderer, .grid__item, .product, .card, article, .product-item")
    if not product_cards:
>>>>>>> 3afbbd23597a33ba8104ff11c6fa0978c4803485:webscraping/scraper.py
        logger.warning(f"No product cards found on page {page_num}.")
        return all_products 

    logger.info(f"Found {len(product_cards)} product cards on page {page_num}")

<<<<<<< HEAD:web_scraping/scraper.py
    for card in product_cards :
        try :
            card_html =str(card )
            logger.debug(f"Card HTML on page {page_num}: {card_html[:200]}...")

            name_el =card.select_one("a.product-title, a.title, a[href][title], a[href]:not([class]), h3,.product__title,.title,.product-title,.product-name, span.title,.product-single__title, div.name")
            name_from_html =name_el.text.strip() if name_el and name_el.text.strip() else(name_el["title"]if name_el and "title"in name_el.attrs else None )

            link_el =card.select_one("a[href]")
            link =link_el["href"]if link_el and link_el.get("href") else "N/A"
            if link !="N/A"and not link.startswith("http"):
                link ="https://eparkville.com"+link 

            name =name_from_html if name_from_html else(urllib.parse.unquote(os.path.basename(link )) if link !="N/A"else "N/A")

            price_el =card.select_one(".price,.money,.price-item--regular,.price--main,.price-amount, span.price")
            price =price_el.text.strip() if price_el else "N/A"
=======
    for card in product_cards:
        try:
            # Log the raw HTML of the card for debugging
            card_html = str(card)
            logger.debug(f"Card HTML on page {page_num}: {card_html[:200]}...")  # Log first 200 chars

            # Extract title with refined selectors (fallback to URL if not found)
            name_el = card.select_one("a.product-title, a.title, a[href][title], a[href]:not([class]), h3, .product__title, .title, .product-title, .product-name, span.title, .product-single__title, div.name")
            name_from_html = name_el.text.strip() if name_el and name_el.text.strip() else (name_el["title"] if name_el and "title" in name_el.attrs else None)

            # Extract link
            link_el = card.select_one("a[href]")
            link = link_el["href"] if link_el and link_el.get("href") else "N/A"
            if link != "N/A" and not link.startswith("http"):
                link = "https://eparkville.com" + link

            # Use last part of URL as name if HTML name is not found
            name = name_from_html if name_from_html else (urllib.parse.unquote(os.path.basename(link)) if link != "N/A" else "N/A")

            # Extract price
            price_el = card.select_one(".price, .money, .price-item--regular, .price--main, .price-amount, span.price")
            price = price_el.text.strip() if price_el else "N/A"
>>>>>>> 3afbbd23597a33ba8104ff11c6fa0978c4803485:webscraping/scraper.py

            all_products.append({
            "name":name,
            "price":price,
            "url":link 
            })
            # Log each product as it's extracted
            logger.info(f"Extracted: Name: {name}, Price: {price}, URL: {link}")
        except Exception as e :
            logger.error(f"Error processing card on page {page_num}: {str(e)}")
            continue 
    return all_products 

<<<<<<< HEAD:web_scraping/scraper.py
def scrape_eparkville():
    all_products =[]
    base_url ="https://eparkville.com/collections/skin-care-products?page="

    for page_num in range(1, 3 ):
        url =f"{base_url}{page_num}"
        logger.info(f"Scraping {url}")

        soup =scrape_with_requests(url )
        if soup :
            products =extract_products(soup, page_num )
            if products :
                all_products.extend(products )
=======
def scrape_eparkville() -> List[Dict[str, str]]:
    """
    Main function to scrape products from eparkville.com throughout multiple pages.

    Returns:
        List[Dict[str, str]]: A consolidated list of all extracted products.
    """
    all_products = []
    base_url = "https://eparkville.com/collections/skin-care-products?page="

    for page_num in range(1, 3):  # Adjusted to 2 pages for testing
        url = f"{base_url}{page_num}"
        logger.info(f"Scraping {url}")

        # First attempt with requests
        soup = scrape_with_requests(url)
        if soup:
            products = extract_products(soup, page_num)
            if products:
                all_products.extend(products)
>>>>>>> 3afbbd23597a33ba8104ff11c6fa0978c4803485:webscraping/scraper.py
                logger.info(f"Extracted {len(products)} products from page {page_num} with requests.")
                continue 

        # Fallback to Selenium if requests fails or no products found
        logger.info(f"Falling back to Selenium for page {page_num}")
        soup =scrape_with_selenium(url )
        if soup :
            products =extract_products(soup, page_num )
            if products :
                all_products.extend(products )
                logger.info(f"Extracted {len(products)} products from page {page_num} with Selenium.")
                continue 

        logger.warning(f"No data extracted from page {page_num}.Saving debug info...")
        if soup :
            with open(f"page_{page_num}_content.html","w", encoding ="utf-8") as f :
                f.write(soup.prettify())

    logger.info(f"Total products extracted: {len(all_products)}")
    return all_products 

<<<<<<< HEAD:web_scraping/scraper.py
def categorize(product ):
    name =product["name"].lower().replace("-"," ")
    if "acne"in name or "blemish"in name or "prone"in name or "acne prone"in name :
=======
def categorize_product(product: Dict[str, str]) -> str:
    """
    Categorize a product based on its name and keywords.

    Args:
        product (Dict[str, str]): The product dictionary containing the name.

    Returns:
        str: The category of the product (e.g., 'Acne', 'Wrinkles', 'Eyebags', 'Eczema', 'Other').
    """
    name = product["name"].lower().replace("-", " ")  # Replace hyphens with spaces for keyword matching
    if "acne" in name or "blemish" in name or "prone" in name or "acne prone" in name:
>>>>>>> 3afbbd23597a33ba8104ff11c6fa0978c4803485:webscraping/scraper.py
        return "Acne"
    elif "wrinkle"in name or "anti aging"in name or "firming"in name or "anti wrinkle"in name :
        return "Wrinkles"
    elif "eye"in name or "under eye"in name or "eye contour"in name :
        return "Eyebags"
    elif "eczema"in name or "soothing"in name or "hydra"in name or "moistur"in name or "hydrating"in name :
        return "Eczema"
    else :
        return "Other"

<<<<<<< HEAD:web_scraping/scraper.py
try :
    products =scrape_eparkville()

    if products :
        logger.info("Raw products extracted:")
        for product in products :
            logger.info(f"Name: {product['name']}, Price: {product['price']}, URL: {product['url']}")
    else :
        logger.warning("No products were extracted before categorization.")

    if products :
        for product in products :
            product["target"]=categorize(product )
        df =pd.DataFrame(products )
        df.to_csv("web_scraping/eparkville_skincare_playwright.csv", index =False, encoding ="utf-8")
        logger.info("Saved all products to eparkville_skincare_playwright.csv")
        logger.info(f"Products by category:\n{df.groupby('target').size()}")
    else :
        logger.warning("No products to save to CSV.")
except Exception as e :
    logger.error(f"Script failed: {str(e)}")
=======
if __name__ == "__main__":
    try:
        products = scrape_eparkville()

        # Log raw products before categorization
        if products:
            logger.info("Raw products extracted:")
            for product in products:
                logger.info(f"Name: {product['name']}, Price: {product['price']}, URL: {product['url']}")
        else:
            logger.warning("No products were extracted before categorization.")

        # Categorize products and save all (including "Other") for debugging
        if products:
            for product in products:
                product["target"] = categorize_product(product)
            df = pd.DataFrame(products)
            # Use relative path for CSV
            csv_path = os.path.join(os.path.dirname(__file__), "eparkville_skincare_playwright.csv")
            df.to_csv(csv_path, index=False, encoding="utf-8")
            logger.info("Saved all products to eparkville_skincare_playwright.csv")
            logger.info(f"Products by category:\n{df.groupby('target').size()}")
        else:
            logger.warning("No products to save to CSV.")
    except Exception as e:
        logger.error(f"Script failed: {str(e)}")
>>>>>>> 3afbbd23597a33ba8104ff11c6fa0978c4803485:webscraping/scraper.py
