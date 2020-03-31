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
```

```python
https://covidtracking.com/api/states/daily
```

```python
https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_json.html
```

```python
r = requests.get('https://covidtracking.com/api/states/daily')
data = r.json()
#pd.json_normalize(data['results'])
```

```python
pd.DataFrame(data)
```
