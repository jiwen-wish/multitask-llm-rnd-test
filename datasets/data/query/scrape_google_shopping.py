from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from tqdm import tqdm 
import argparse
import json
import pandas as pd
import time

parser = argparse.ArgumentParser()
parser.add_argument('--input')
parser.add_argument('--input_key')
parser.add_argument('--output')
args = parser.parse_args()

class Scraper:
    def __init__(self, output, driver='chrome', driverpath='/usr/bin/chromedriver'):
        if driver == 'chrome':
            options = webdriver.ChromeOptions()
            options.add_argument('--headless')
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            self.driver = webdriver.Chrome(driverpath, chrome_options=options)
        else:
            raise Exception(f'no driver: {driver}')
        self.output = output
        self.scraped_queries = set()
        try:
            with open(output, 'r') as f:
                for i in f:
                    self.scraped_queries.add(json.loads(i)['query'])
        except Exception as e:
            print(e)
        print(f'existing {len(self.scraped_queries)} queries')

    def scrape(self, queries_):
        """Scrape Google Shopping's products given a list of search queries and store product titles in a list"""
        queries = list(set(queries_) - self.scraped_queries)
        print(f'scrape {len(queries)} queries out of input {len(queries_)} queries after dedup')
        self.driver.get('https://www.google.com/shopping')
        self.driver.implicitly_wait(5)
        nlines= 0
        with open(self.output, 'a') as f:
            for query in tqdm(queries):
                try:
                    print(f'Scrape {query}')
                    self.driver.find_element(By.NAME, 'q').send_keys(query)
                    self.driver.find_element(By.NAME, 'q').send_keys(Keys.RETURN)
                    self.driver.implicitly_wait(5)
                    product_elements = self.driver.find_elements(By.CLASS_NAME, 'sh-dgr__content')
                    out_i = None
                    for product_element in product_elements:
                        # parse out h3 in product_element, which contains title
                        try:
                            title_element = product_element.find_element(By.TAG_NAME, 'h3')
                            title = title_element.text
                            url = product_element.find_element(By.TAG_NAME, 'a').get_attribute('href')
                            out_i = {'query': query, 'title': title, 'url': url}
                            f.write(json.dumps(out_i))
                            f.write('\n')
                            nlines += 1
                        except Exception as e:
                            pass
                    if out_i is not None:
                        self.scraped_queries.add(query)
                    # clear searchbar
                    self.driver.find_element(By.NAME, 'q').clear()
                except Exception as e:
                    print(f'Failed to scrape {query} due to {e}')
                    time.sleep(60 * 5)
        print(f"Added {nlines} lines to {self.output}")
        self.driver.close()


if __name__ == '__main__':
    df_queries = pd.read_json(args.input, lines=True)
    scraper = Scraper(args.output)
    queries = df_queries[args.input_key].tolist()
    scraper.scrape(queries)