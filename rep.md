# Using Python with Stata

##  [Hua Peng@StataCorp][hpeng]
### 2019 Stata User Conference Chicago 
### [https://huapeng01016.github.io/chicago19/](https://huapeng01016.github.io/chicago19/)

# Stata 16 has tight integration with Python

- embed and execute Python code 
- use Python interactively
- define and use Python routines in do-files and ado-files
- Stata Function Interface (sfi)

# First interactive session

~~~~
<<dd_do>>
python:
2+3
print(2*3)
end
<</dd_do>>
~~~~

# A more interesting interactive session

~~~~
<<dd_do>>
python:
from functools import reduce
a = [1, 2, 3, 4, 5, 6, 7] 
sum = reduce((lambda x, y: x + y), a)
print(sum) 
end
<</dd_do>>
~~~~

# Use SFI 

~~~~
<<dd_do>>
python:
from functools import reduce
from sfi import Data
stata: sysuse auto, clear
a = Data.get(var='price') 
sum = reduce((lambda x, y: x + y), a)
from sfi import Macro
Macro.setLocal('sum', str(sum))
end
di "sum of price = `sum'"
<</dd_do>>
~~~~

# Another example

## Setup 

~~~~
<<dd_do>>
python:
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from sfi import Data
end
<</dd_do>>
~~~~

## Get data from Stata
~~~~
<<dd_do>>
python:
stata: sysuse sandstone, clear
D = np.array(Data.get("northing easting depth"))
end
<</dd_do>>
~~~~

## Graph
~~~~
python:
ax = plt.axes(projection='3d')
plt.xticks(np.arange(60000, 90001, step=10000))
plt.yticks(np.arange(30000, 50001, step=5000))
ax.plot_trisurf(D[:,0], D[:,1], D[:,2], cmap='viridis', edgecolor='none');
plt.savefig("sandstone.png")
end
~~~~

<<dd_do:quietly>>
python:
ax = plt.axes(projection='3d')
plt.xticks(np.arange(60000, 90001, step=10000))
plt.yticks(np.arange(30000, 50001, step=5000))
ax.plot_trisurf(D[:,0], D[:,1], D[:,2], cmap='viridis', edgecolor='none');
plt.savefig("sandstone.png")
end
<</dd_do>>

## Output
![sandstone image](sandstone.png "sandstone.png")

# Useful tips

- Python environment
- Indentation and space
- Package management

# Web scraping 

## Use **pandas** to get Nasdaq 100 ticks

~~~~
python:
import pandas as pd

data = pd.read_html("https://en.wikipedia.org/wiki/NASDAQ-100")
table = data[2]
t1 = table[1]
t1 = t1[1:].tolist()
end
~~~~

## Put results into a Stata varaible

~~~~
python:
from sfi import Data
Data.addObs(len(t1))
stata: gen ticker = ""
Data.store('ticker', range(len(t1)), t1)
end
~~~~

## Resulted dataset

<<dd_do: quietly>>
use stata/nasdaq_tic.dta, clear
<</dd_do>>

~~~~
<<dd_do>>
list in 1/5, clean
<</dd_do>>
~~~~

## Get 

# Support Vector Machine (SVM)

# Recap

# Thanks!

[hpeng]: hpeng@stata.com
