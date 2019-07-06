# Using Python with Stata

##  [Hua Peng@StataCorp][hpeng]
### 2019 Stata User Conference Chicago 
### [https://tinyurl.com/y3h35tt3](https://huapeng01016.github.io/chicago19/)

# Stata 16 introduces tight integration with Python

- Embed and execute Python code 
- Use Python interactively
- Define and use Python routines in do-files and ado-files
- Inteact with Stata through Stata Function Interface (sfi)

# Use Python i0nteractively

## Interactive session 1

**Hello World!**

~~~~
<<dd_do>>
python:
print('Hello World!')
2+3
end
<</dd_do>>
~~~~

## Interactive session 2

Indentation is important when typing Python code in Stata. 

~~~~
<<dd_do>>
python:
sum = 0
for i in range(7):
    sum = sum + i 
print(sum) 
end
<</dd_do>>
~~~~

## Interactive session 3

**sfi** allows Python code to bidirectionally communicate with Stata.

~~~~
<<dd_do>>
python:
from functools import reduce
from sfi import Data, Macro
stata: quietly sysuse auto, clear
sum = reduce((lambda x, y: x + y), Data.get(var='price'))
Macro.setLocal('sum', str(sum))
end
display "sum of var price is : `sum'"
<</dd_do>>
~~~~

## Interactive session 4

Use **sfi** to handle missing values.

~~~~
<<dd_do>>
python:
stata: quietly sysuse auto, clear
sum1 = reduce((lambda x, y: x + y), Data.get(var='rep78'))
sum1
sum2 = reduce((lambda x, y: x + y), Data.get(var='rep78', selectvar=-1))
sum2
end
<</dd_do>>
~~~~

# Use Python packages

* Pandas
* Numpy, PuLP, PyMC3
* lxml, BeautifulSoup
* Matplotlib, ggplot2
* scikit-learn, Tensorflow, Keras, PyTorch
* NLTK, spacy

# A 3d surface plot example

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

We use **sandstone** dataset, which is the example dataset for **twoway contour**.

~~~~
<<dd_do>>
python:
stata: sysuse sandstone, clear
D = np.array(Data.get("northing easting depth"))
end
<</dd_do>>
~~~~

## Graph

We draw a surface plot using a triangular mesh. 

~~~~
python:
ax = plt.axes(projection='3d')
plt.xticks(np.arange(60000, 90001, step=10000))
plt.yticks(np.arange(30000, 50001, step=5000))
ax.plot_trisurf(D[:,0], D[:,1], D[:,2], cmap='viridis', edgecolor='none')
plt.savefig("sandstone.png")
end
~~~~

<<dd_do:quietly>>
python:
ax = plt.axes(projection='3d')
plt.xticks(np.arange(60000, 90001, step=10000))
plt.yticks(np.arange(30000, 50001, step=5000))
ax.plot_trisurf(D[:,0], D[:,1], D[:,2], cmap='viridis', edgecolor='none')
plt.savefig("sandstone.png")
end
<</dd_do>>

## Output
![sandstone image](sandstone.png "sandstone.png")

## Change the look of the graph

~~~~
python:
ax.plot_trisurf(D[:,0], D[:,1], D[:,2], cmap=plt.cm.Spectral, edgecolor='none')
plt.savefig("sandstone1.png")
end
~~~~

<<dd_do:quietly>>
python:
ax.plot_trisurf(D[:,0], D[:,1], D[:,2], cmap=plt.cm.Spectral, edgecolor='none')
ax.view_init(30, 60)
plt.savefig("sandstone1.png")
end
<</dd_do>>

## Output
![sandstone1 image](sandstone1.png "sandstone1.png")

## Or make some animation ([do-file](./stata/gif3d.do))

![sandstone gif](./sandstone.gif)

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

# Useful tips

- Python environment
- Package management

# Thanks!

[hpeng]: hpeng@stata.com
