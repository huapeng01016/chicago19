cscript

python:
import pandas as pd

data = pd.read_html("https://en.wikipedia.org/wiki/NASDAQ-100")
table = data[2]
t1 = table[1]
t1 = t1[1:].tolist()
t1

from sfi import Data
Data.addObs(len(t1))
stata: gen ticker = ""
Data.store('ticker', range(len(t1)), t1)
end

list


