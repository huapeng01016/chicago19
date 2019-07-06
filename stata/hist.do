cscript 

python:
import pandas as pd

data = pd.read_html("https://www.nasdaq.com/symbol/aapl/historical")
len(data)
table = data[2]
table
t1 = table[1:]
t1
Data.addObs(63)
stata: gen date   = ""
stata: gen open   = .
stata: gen high   = .
stata: gen low    = .
stata: gen close  = .
stata: gen double volumn = .

Data.store(None, range(63), t1.values.tolist())
end
