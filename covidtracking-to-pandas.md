---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.3.1
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

```python
import pandas as pd
import numpy as np
import seaborn as sns
import requests
import json

%reload_ext autoreload
%autoreload 2

import matplotlib as mpl
import matplotlib.pyplot as plt
import pytoolsMH as ptMH
import seaborn as sns
import os,sys
import scipy.io
import scipy.stats as ss
from pathlib import Path
import statsmodels.api as sm
import statsmodels.formula.api as smf

sns.set_style('whitegrid')

sys.path.append('../src')

import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_rows', None)

mpl.rc('pdf', fonttype=42) # embed fonts on pdf output 

r_ = np.r_
```

https://covidtracking.com/api/states/daily


https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_json.html

```python
r = requests.get('https://covidtracking.com/api/states/daily')
data = r.json()
#pd.json_normalize(data['results'])
```

```python

```

```python

```

```python
df = pd.DataFrame(data)
df = df.fillna(0)
```

```python
MD = df[df['state'] == 'MD']
DC = df[df['state'] == 'DC']
VA = df[df['state'] == 'VA']
```

```python
def resetdays(df):
    df = df.sort_values('date')
    df = df[df['death']!= 0]
    df = df.reset_index(drop = True)
    df = df.reset_index()
    df = df.rename(columns={"index": "days"})
    return df
```

```python
MD = resetdays(MD)
VA = resetdays(VA)
DC = resetdays(DC)
```

```python
DMV = pd.concat([MD, DC, VA])
```

```python
sns.lineplot(x = 'days', y = 'death', data = DMV, hue = 'state', 
             palette = 'magma_r', style = 'state', lw = 2)
plt.yscale('log')
plt.yticks([1, 2, 5, 10, 20, 50], 
           [1, 2, 5, 10, 20, 50], fontsize = 10)
plt.ylabel('Deaths', fontsize = 15)
plt.xlabel('Days since first death', fontsize = 15)
plt.title('Coronavirus Deaths in DC, MD, and VA', fontsize = 20)
plt.legend(fontsize = 10)
```

```python

```

```python

```
