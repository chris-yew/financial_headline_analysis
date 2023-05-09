from bs4 import BeautifulSoup
import requests


def get_headlines(num):
    res = []
    url = 'https://sg.finance.yahoo.com/topic/economy/'
    page = requests.get(url)
    soup = BeautifulSoup(page.content, "html.parser")

    target_class = 'Mb(5px)'
    results = soup.find_all(class_=target_class)

    for i in range(num):
        res.append(results[i].text)
    return res


if __name__ == '__main__':
    print(get_headlines(10))
