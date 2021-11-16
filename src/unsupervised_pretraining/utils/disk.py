import requests
from urllib.parse import urlencode

BASE_URL = "https://cloud-api.yandex.net/v1/disk/public/resources/download?"


def load_file(url):
    url = BASE_URL + urlencode(dict(public_key=url))
    response = requests.get(url)
    download_url = response.json()["href"]

    file = requests.get(download_url)

    return file.content
