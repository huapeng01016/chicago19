python:
import bs4 as bs
import pickle
import requests
import base64
import csv
 
def save_sp500_tickers(file):
    resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(resp.text, 'lxml')
    table = soup.find('table', {'class': 'wikitable sortable'})
    tickers = []
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text
        tickers.append(ticker.strip())
    
    with open(file, 'w') as csv_file:
		wr = csv.writer(csv_file)
		wr.writerow(tickers)
		
    return tickers
end

python:save_sp500_tickers("test.csv")
