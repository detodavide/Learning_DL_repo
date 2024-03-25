from dotenv import load_dotenv, find_dotenv
import os
from selenium.webdriver import Chrome, ChromeOptions
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import StaleElementReferenceException, TimeoutException
import undetected_chromedriver as uc
from webdriver_manager.chrome import ChromeDriverManager
import requests

load_dotenv(find_dotenv())
BASE_URL = "https://python.langchain.com/docs/get_started/introduction"
BRAVE_DATA_DIR = os.getenv("BRAVE_DATA_DIR")


def get_destination_path(filename, out_data_type="txt"):
    parent_path = os.getcwd()
    docs_path = "/raw_data/"
    docs_path = docs_path + f"csv/" if out_data_type == "csv" else docs_path + f"txt/"
    print(docs_path)
    return parent_path + docs_path + filename


def extract_descendant_links(element):
    """
    Recursively extract all href links from descendants of the given element.
    """
    # Find all descendant elements of the current element
    descendant_elements = element.find_elements(By.XPATH, ".//*")
    for descendant in descendant_elements:
        # If the descendant is an anchor element, extract its href attribute
        try:
            href = descendant.get_attribute("href")
            if href:
                print("Anchor href:", href)
        except Exception as e:
            print(e)

        tag_name = descendant.tag_name
        text = descendant.text

        print("Tag name:", tag_name)
        print("Text:", text)

        # Recursively extract links from descendants of the current descendant
        extract_descendant_links(descendant)


def build_driver(env_dir, brave_browser=False):
    options = uc.ChromeOptions()
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--disable-browser-side-navigation")
    options.add_argument(rf"--user-data-dir={env_dir}")
    if brave_browser:
        options.binary_location = "/usr/bin/brave-browser"
    options.add_argument("--disable-extensions")
    options.add_argument("--dns-prefetch-disable")
    options.add_argument("--disable-gpu")
    options.add_argument("--disable-application-cache")
    options.add_argument("--disable-cache")
    options.add_argument("--disable-offline-load-stale-cache")
    options.add_argument("--disable-software-rasterizer")
    driver = uc.Chrome(
        driver_executable_path=ChromeDriverManager().install(), options=options
    )

    driver.set_page_load_timeout(10)

    return driver


def extractor():

    driver = build_driver(BRAVE_DATA_DIR, brave_browser=True)
    driver.get(BASE_URL)
    file_path = get_destination_path(filename="langchain_text1.txt")

    outer_list_items = WebDriverWait(driver, 5).until(
        EC.presence_of_all_elements_located(
            (By.XPATH, "/html/body/div/div[2]/div/aside/div/div/nav/ul/li")
        )
    )

    for li in outer_list_items:
        # Recursively extract links from descendants of the current list item
        extract_descendant_links(li)

    # with open(file_path, "w", encoding="utf-8") as file:
    #     file.write(text_data)

    # print(f"Text data saved to '{file_path}' successfully.")


if __name__ == "__main__":
    extractor()
