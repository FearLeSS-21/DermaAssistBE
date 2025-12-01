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

# Configure logging to file and console with UTF-8 encoding
logging.basicConfig(
    level=logging.DEBUG,  # DEBUG level to see card HTML
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("scraper_log.txt", encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger()

# Force UTF-8 encoding for console output
sys.stdout.reconfigure(encoding="utf-8")

def scrape_with_requests(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        return soup
    except requests.RequestException as e:
        logger.error(f"Request failed for {url}: {str(e)}")
        return None

def scrape_with_selenium(url):
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)

    try:
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
    finally:
        driver.quit()

def extract_products(soup, page_num):
    all_products = []
    if not soup:
        return all_products

    # Try common Shopify product card structures
    product_cards = soup.select(".product-card-renderer, .grid__item, .product, .card, article, .product-item")
    if not product_cards:
        logger.warning(f"No product cards found on page {page_num}.")
        return all_products

    logger.info(f"Found {len(product_cards)} product cards on page {page_num}")

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

            all_products.append({
                "name": name,
                "price": price,
                "url": link
            })
            # Log each product as it's extracted
            logger.info(f"Extracted: Name: {name}, Price: {price}, URL: {link}")
        except Exception as e:
            logger.error(f"Error processing card on page {page_num}: {str(e)}")
            continue
    return all_products

def scrape_eparkville():
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
                logger.info(f"Extracted {len(products)} products from page {page_num} with requests.")
                continue

        # Fallback to Selenium if requests fails or no products found
        logger.info(f"Falling back to Selenium for page {page_num}")
        soup = scrape_with_selenium(url)
        if soup:
            products = extract_products(soup, page_num)
            if products:
                all_products.extend(products)
                logger.info(f"Extracted {len(products)} products from page {page_num} with Selenium.")
                continue

        logger.warning(f"No data extracted from page {page_num}. Saving debug info...")
        if soup:
            with open(f"page_{page_num}_content.html", "w", encoding="utf-8") as f:
                f.write(soup.prettify())

    logger.info(f"Total products extracted: {len(all_products)}")
    return all_products

# Categorization logic with expanded keywords
def categorize(product):
    name = product["name"].lower().replace("-", " ")  # Replace hyphens with spaces for keyword matching
    if "acne" in name or "blemish" in name or "prone" in name or "acne prone" in name:
        return "Acne"
    elif "wrinkle" in name or "anti aging" in name or "firming" in name or "anti wrinkle" in name:
        return "Wrinkles"
    elif "eye" in name or "under eye" in name or "eye contour" in name:
        return "Eyebags"
    elif "eczema" in name or "soothing" in name or "hydra" in name or "moistur" in name or "hydrating" in name:
        return "Eczema"
    else:
        return "Other"

# Run the scraper
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
            product["target"] = categorize(product)
        df = pd.DataFrame(products)
        df.to_csv("/home/GradProject2025NU/GProject/webscrappinng/eparkville_skincare_playwright.csv", index=False, encoding="utf-8")
        logger.info("Saved all products to eparkville_skincare_playwright.csv")
        logger.info(f"Products by category:\n{df.groupby('target').size()}")
    else:
        logger.warning("No products to save to CSV.")
except Exception as e:
    logger.error(f"Script failed: {str(e)}")