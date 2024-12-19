"""

Utility functions for the WebsiteCatergorization module.

"""

from string import punctuation
from requests import get
from bs4 import BeautifulSoup
from nltk.corpus import stopwords

def get_bytes_from_url(url: str, timeout: int = 10) -> bytes:

    """
    Get the bytes from a URL.

    Args:
        url (str): The URL to get the bytes from.
        timeout (int, optional): The timeout for the request. Defaults to 10.

    Returns:
        bytes: The bytes from the URL.
    """

    response = get(url, timeout=timeout)

    response.raise_for_status()

    return response.content

def get_hyperlinks_from_page_bytes(page_bytes: bytes) -> list[str]:

    """
    Get the hyperlinks from a page.

    Args:
        page_bytes (bytes): The bytes of the page.

    Returns:
        list[str]: The hyperlinks from the page.
    """

    soup = BeautifulSoup(page_bytes, "html.parser")

    hyperlinks = []

    link: BeautifulSoup = None

    for link in soup.find_all("a"):
        hyperlinks.append(link.get("href", default=None))

    return [i for i in hyperlinks if i is not None]

def clean_html(page_bytes: bytes) -> str:

    """
    Clean the HTML of a page (in bytes).

    Args:
        page_bytes (bytes): The bytes of the page.

    Returns:
        str: The cleaned HTML.
    """

    soup = BeautifulSoup(page_bytes, "html.parser")

    for data in soup(['style', 'script', 'code', 'a']):

        data.decompose()

    return ' '.join(soup.stripped_strings)

test = get_bytes_from_url("https://pulse.zerodha.com")

def clean_text(text: str) -> str:

    """
    Clean the text of a page (convert to lowercase, remove stopwords, remove punctuation).

    Args:
        text (str): The text to clean.

    Returns:
        str: The cleaned text.
    """

    #convert text to lowercase
    text = text.lower()

    #remove newlines
    text = text.replace("\n", " ")

    #remove stopwords
    temp_stopwords = set(stopwords.words("english"))
    text = [word for word in text.split() if word not in temp_stopwords]

    #remove whitespace
    text = [word for word in text if word.strip()]

    #remove punctuation
    translator = str.maketrans('', '', punctuation+"—")
    text = " ".join(text).translate(translator)

    return text

test = get_bytes_from_url("https://pulse.zerodha.com")
#print(clean_html(test))
print(clean_text(clean_html(test)))
