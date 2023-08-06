"""This is my recreation of an existing library that did not work for me"""
"""but I was able to recreate the functions with some minor tweaks.  The """
"""original module is named 'finvizfinance' and is associated with 'finviz.com'"""

import sys
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

headers = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_4) \
            AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.138 Safari/537.36"
}

session = requests.Session()




def web_scrap(stck):
    """Scrap website.
    
    """

    QUOTE_URL = "https://finviz.com/quote.ashx?t={ticker}"
    quote_urla = QUOTE_URL.format(ticker = stck)


    try:
        website = session.get(quote_urla, headers=headers, timeout=10)
        website.raise_for_status()
        soup = BeautifulSoup(website.text, "lxml")
    except requests.exceptions.HTTPError as err:
        raise Exception(err)
    except requests.exceptions.Timeout as err:
        raise Exception(err)
    return soup