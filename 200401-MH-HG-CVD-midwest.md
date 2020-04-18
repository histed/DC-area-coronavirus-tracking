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

# Load covidtracking data and make some plots

- 4/6 todo
  - set arrow location automatically (xmax)
  
- 4/5 todo
  - [X] hannah refactored code, I adapted with recent updates
  - [X] run code tonight
  - [ ] fix death inset not being annotated
  

```python

```

```python
%reload_ext autoreload
%autoreload 2

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pytoolsMH as ptMH
import pandas as pd
import seaborn as sns
import os,sys
import scipy.io
import scipy.stats as ss
from pathlib import Path
import statsmodels.api as sm
import statsmodels.formula.api as smf
import requests
import json
import datetime, dateutil.parser
from argparse import Namespace

sns.set_style('whitegrid')

from src import covid_midwest as cvd

import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)

mpl.rc('pdf', fonttype=42) # embed fonts on pdf output 

r_ = np.r_
```

## covidtracking.com data

```python
r = requests.get('https://covidtracking.com/api/states/daily')
data = r.json()
ctDf = pd.DataFrame(data)

# save current data
datestr = datetime.datetime.now().strftime('%y%m%d')
ctDf.to_hdf('./ct-data/covidtracking-data-%s.h5'%datestr, key='ct', complevel=9, complib='zlib')
```

```python
paramsC = pd.DataFrame(index={'WI','IL','MN','NY'}, columns=['fullname'])
paramsC.loc[:,'fullname'] = pd.Series({ 'WI': 'Wisconsin', 'IL': 'Illinois', 'MN': 'Minnesota'})
paramsC.loc[:,'labYOff'] = pd.Series({ 'WI': -15, 'IL': +10, 'MN': -10, 'NY':-15})
paramsC.loc[:,'labXOff'] = pd.Series({ 'WI': 0, 'IL': 0, 'MN': +5, 'NY': 0})
paramsC.loc[:,'lw'] = pd.Series({ 'WI': 2, 'IL': 2, 'MN': 2, 'NY': 0.8})
#params.loc[:,'xoff'] = pd.Series({ 'DC': -9, 'MD': -6, 'VA': -6, 'NY': -0.3})
paramsC.loc[:,'xoff'] = pd.Series({ 'WI': 0, 'IL': 0, 'MN': 0, 'NY': -1})
paramsC.loc[:,'color'] = pd.Series({ 'NY': 0.4, 'WI': '#1f77b4', 'IL': '#ff7f0e', 'MN': '#2ca02c'})

display(paramsC)
```

```python
#Data: http://covidtracking.com  updated Apr 04 2020, 20:00 EDTGraphic: Hannah Goldbach, Mark Histed  @hannah_goldbach @histedlab
tDStr = datetime.date.today().strftime('%b %-d 2020')
tCredStr = 'Updated %s, 20:00 EDT\n  data: http://covidtracking.com\nGraphic: Hannah Goldbach, Mark Histed\n  @hannah_goldbach @histedlab' % tDStr
print(tCredStr)
```

```python
sns.set_style('darkgrid')
fig, ax = plt.subplots(figsize=r_[1,1]*6, dpi=120)

xlim = r_[0,45]
todayx = 0 #26

# big plot - states
df = ctDf.copy()
for st in ['WI', 'IL', 'MN', 'NY']:
    df, paramsC = cvd.plot_state(df, st, paramsC, ax, False)

# big plot fixup
cvd.fixups(ax)
#ADJUST AXIS LIMS TO FIT 
ax.set_xlim(xlim)  
ax.set_ylim([10, ax.get_ylim()[1]])
ax.set_ylabel('Total cases')

cvd.plot_guide_lines(ax)

# inset
ylim = r_[600, 10000]*2.3 #ADJUST YLIM TO FIT 
axins = cvd.inset(df, paramsC, ax, ylim, is_inset=True)

#case doubling lines 
xs = r_[1,10] #ADJUST COORDS AS CASES CLIMB
dtL = [2,3,4]
cvd.case_anno_inset_double(xs, axins, paramsC) #might have to adjust in scropt 

#add arrow
tStr = datetime.date.today().strftime('%B %-d')
ax.annotate(tStr, xy=(41,10), xycoords='data', xytext=(0,-30), textcoords='offset points',
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3', color='0.3'),
            color='0.3', ha='center')

## credit string
#ax.annotate(tCredStr, xy=(0.9,0.1), xycoords='axes fraction')

# save fig 
fig.tight_layout()
fig.savefig('./fig-output/midwest-ct-%s.png'%datestr, dpi=300, bbox_inches='tight', pad_inches=0.5)
            #bbox_inches=r_[0,0,10,15])#, 
```

### Same plot for deaths

```python
paramsD = paramsC.copy()
#pd.DataFrame(index={'DC','MD','VA','NJ'}, columns=['fullname'])
#paramsD.loc[:,'fullname'] = pd.Series({ 'DC': 'District of Columbia', 'MD': 'Maryland', 'VA': 'Virginia'})
#paramsD.loc[:,'labYOff'] = pd.Series({ 'DC': +10, 'MD': -10, 'VA': +15, 'NJ':0})
#paramsD.loc[:,'labXOff'] = pd.Series({ 'DC': 0, 'MD': +5, 'VA': +5, 'NJ': +10})
#paramsD.loc[:,'lw'] = pd.Series({ 'DC': 2, 'MD': 2, 'VA': 2, 'NJ': 0.8})
#paramsD.loc[:,'xoff'] = pd.Series({ 'DC': 0, 'MD': 0, 'VA': 0, 'NJ': -1})
```

```python
sns.set_style('darkgrid')
fig, ax = plt.subplots(figsize=r_[1,1]*6, dpi=120)

xlim = r_[0,45]

todayx = 0 #26

#df = ctDf.copy()
for st in ['WI', 'IL', 'MN', 'NY']:
    df, paramsD = cvd.plot_state(df, st, paramsD, ax, False, is_cases=False)

# big plot fixup
cvd.fixups(ax)
#ADJUST AXIS LIMS TO FIT 
ax.set_xlim(xlim)  
ax.set_ylim([1, ax.get_ylim()[1]])
ax.set_ylabel('Total deaths')
cvd.plot_guide_lines(ax, yoffset_mult=10**-1.5)

# inset
ylim = r_[9, 50]*1.3 #ADJUST YLIM TO FIT  - get aspect ratio right
#cvd.inset(df, paramsD, ax, ylim, is_inset=True, is_cases=False)


#Death doubling annotation
#cvd.case_anno_inset_double(xs, axins, paramsD) #might have to adjust in scropt 


#add arrow
tStr = datetime.date.today().strftime('%B %-d')
ax.annotate(tStr, xy=(41, 1), xycoords='data', xytext=(0,-30), textcoords='offset points',
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3', color='0.3'), 
            color='0.3', ha='center')

fig.tight_layout()
fig.savefig('./fig-output/midwest-dt-%s.png'%datestr, dpi=300, bbox_inches='tight', pad_inches=0.5)
            #bbox_inches=r_[0,0,10,15])#, 
```

```python
for state in ['DC']:
    desIx = ctDf.state == state
    stDf = ctDf.loc[desIx,:].copy()
    stDf.set_index('date', inplace=True)

    posV = stDf.loc[:,'positive'][::-1]
```

```python
MNnew = pd.DataFrame({'date':[20200305, 20200304], 'state':['MN', 'MN']})
ctDf2 = pd.concat([ctDf, MNnew])
```

# Positive test rates in MD, DC, VA

```python
sns.set_style('darkgrid')

tDStr = datetime.date.today().strftime('%B %-d')
daylabel = f'Days - last is {tDStr}'

fig = plt.figure(figsize=r_[1,0.75]*[2,3]*5, dpi=100)
gs = mpl.gridspec.GridSpec(3,2)

ax = plt.subplot(gs[0,0])
datD = {} 
ctDf2 = pd.concat([ctDf, MNnew])
for state in ['WI', 'IL', 'MN']:
    desIx = ctDf2.state == state
    stDf = ctDf2.loc[desIx,:].copy()
    stDf.set_index('date', inplace=True)

    posV = stDf.loc[:,'positive'][::-1]
    negV = stDf.loc[:,'negative'][::-1]
    pdV = np.diff(posV)
    ndV = np.diff(negV)
    # manual adjustments
    pctPos = pdV/(pdV+ndV)*100        
    datD[state] = Namespace(posV=posV, negV=negV, pdV=pdV, ndV=ndV, pctPos=pctPos)
    plt.plot(pctPos, '.-', label=state)
ax.set_title('Percent of tests positive, WI area')    
#ax.legend(loc=2)
ax.set_ylabel('Positive tests per day (%)')
ax.set_xlabel(daylabel)
# markup line
maxN = len(datD['IL'].pctPos)
meanNDay = 7
desNs = r_[maxN-meanNDay:maxN]
tV = np.hstack([datD[x].pctPos[desNs] for x in datD.keys()])
tM = np.mean(tV)
plt.plot(desNs, tM+desNs*0, color='k', lw=5, ls='-', alpha=0.5)
# anno it
ax.annotate('mean\npos test rates,\nlast %d days'%meanNDay, 
            xy=(desNs[2],tM), xycoords='data', xytext=(-50,-120), textcoords='offset points',
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.1', color='k'), 
            color='0.3', ha='center')

ax = plt.subplot(gs[0,1])
ax2 = plt.subplot(gs[1,1])
ax3 = plt.subplot(gs[2,1])
for state in ['WI', 'IL', 'MN']:
    dd = datD[state]
    display(len(dd.posV))
    pH = ax.plot(dd.pdV, '.-', label=state)        
    pH = ax2.plot(dd.ndV, '.-', label=state)            
    pH = ax3.plot((dd.ndV+dd.pdV), '.-', label=state)
ax.legend()
ax.set_title('Positive results per day')
ax.set_ylabel('Results')

ax2.set_title('Negative results per day')
ax2.set_ylabel('Results')

ax3.set_title('Total results per day')
ax3.set_ylabel('Results')
ax3.set_xlabel(daylabel)

tStr = datetime.date.today().strftime('%a %B %-d')
fig.suptitle(f'{tStr}: \n'
             'Test rates, Wisconsin Area (WI, IL, MN)',
             fontsize=16, fontname='Roboto', fontweight='light', 
             x=0.05, y=0.92, ha='left', va='bottom')


doSave = True
if doSave:
    fig.savefig('./fig-output/midwest-testing-%s.png'%datestr, 
            dpi=300, bbox_inches='tight', pad_inches=0.5)
            #bbox_inches=r_[0,0,10,15])#, 


```

### Doubling time plot

```python
def double_time(st, doubles):
    dD = paramsC.loc[st, 'plot_data']
    slope0 = np.diff(np.log10(dD['ys'].to_numpy()))
    double_time = -np.log10(2)/slope0
    if st == 'MN': 
        double_time = np.append(double_time, [0, 0])
    doubles[st] = double_time
    return(doubles)

'''def pct_rise(st, outDf):
    dD = paramsC.loc[st, 'plot_data']
    pctsTemp = pd.DataFrame()
    for tSt in np.arange(0, len(dD['xs'])-2, 1):
        ns = r_[0:2]+tSt
        x0 = np.mean(dD['xs'][ns])
        y0 = np.mean(dD['ys'].iloc[ns])
        slope0 = np.log10(dD['ys'].iloc[ns[0]])-np.log10(dD['ys'].iloc[ns[1]])
        pct_rise = (dD['ys'].iloc[ns[0]]/dD['ys'].iloc[ns[1]] * 100) - 100
        pctsTemp.append(pct_rise)
        if st == 'MN': 
            pctsTemp = np.append(pctsTemp, [0, 0])
    pcts[st] = pctsTemp
    return pcts'''

def find_days(df): 
    df = df.reindex(index=df.index[::-1])
    df = df.reset_index(drop = True)
    df = df.reset_index()
    df = df.rename(columns = {'index': 'day'})
    return df
```

```python
doubles = pd.DataFrame(columns = {'WI', 'IL', 'MN'})
#pcts = pd.DataFrame(columns = {'WI', 'IL', 'MN'})

for st in ['WI', 'IL', 'MN']:
    doubles = double_time(st, doubles)
    #pcts = pct_rise(st, pcts)

doubles = find_days(doubles)
#pcts = find_days(pcts)

dtV = paramsC.loc['WI', 'plot_data']['dtV']
```

```python
doubles = doubles.replace([np.inf, -np.inf], np.nan)
for st in ['WI', 'IL', 'MN']:
    doubles[st].fillna((doubles[st].mean()), inplace=True)
```

```python
import pytoolsMH as ptMH
for st in ['WI', 'IL', 'MN']:
    doubles[st+'_smooth'] = ptMH.math.smooth_lowess(doubles[st], x=None, span=7, robust=False, iter=None, axis=-1)
```

```python
fig = plt.figure(figsize=r_[4, 3], dpi=100)
for st in ['WI', 'IL', 'MN']:
    plt.plot(doubles['day'], doubles[st], alpha = 0.8, lw = 0.75)
    plt.legend()
plt.plot(doubles['day'], doubles['WI_smooth'], label = 'WI', lw = 2.5, color = '#2678B2')
plt.plot(doubles['day'], doubles['MN_smooth'], label = 'MN', lw = 2.5, color = '#339F34')
plt.plot(doubles['day'], doubles['IL_smooth'], label = 'IL', lw = 2.5, color = '#FD7F28')
plt.legend()
plt.xlabel('days since first test')
plt.ylabel('time for cases to double (days)')

fig.savefig('./fig-output/midwest-doubling-%s.png'%datestr, 
            dpi=300, bbox_inches='tight', pad_inches=0.5)

"""plt.axvline(x = 30, alpha = 0.5)
plt.axvline(x = 23, alpha = 0.5)
plt.axvline(x = 16, alpha = 0.5)
plt.axvline(x = 9, alpha = 0.5)"""
```

## MH figure

```python
doSave = True

for st in ['WI', 'IL', 'MN']:
    doubles[st+'_smooth'] = ptMH.math.smooth_lowess(doubles[st], x=None, span=8, robust=False, iter=None, axis=-1)

sns.set_style('whitegrid')
fig = plt.figure(figsize=r_[4, 3]*1.5, dpi=100)
ax = plt.subplot()
ax.set_facecolor('#f6fcfd')


for (iS,st) in enumerate(['WI', 'IL', 'MN']):
    pH, = plt.plot(doubles['day'], doubles[st], alpha = 0.8, lw = 0.75)
    plt.plot(doubles['day'], doubles[st+'_smooth'], label = st, lw = 2.5, color = pH.get_color())
    last_double = doubles[st+'_smooth'].iloc[-1]
    paramsC.loc[st, 'last_double'] = last_double
    paramsC.loc[st, 'color'] = pH.get_color()

# last_double annotate    
for (iS,st) in enumerate(['WI', 'IL', 'MN']):
    last_double = paramsC.loc[st, 'last_double']
    xy = (1.05, 0.65-iS*0.08)
    tStr = f'{st}: {last_double:.2g}'
    if iS == 0:
        tStr = tStr + ' days'
    ax.annotate(tStr, xy=xy, 
                xycoords='axes fraction', color=paramsC.loc[st, 'color'], ha='left',
                fontweight='bold', fontsize=14)
    
#plt.legend()
plt.ylabel('doubling time for cases (days)', fontsize=12)

plt.grid(False, which='both', axis='x')
sns.despine(left=True, right=True, top=False, bottom=False)

x_dates = dtV.dt.strftime('%b %-d')
xt = r_[5:len(x_dates):7]
ax.set_xticks(xt-1)
ax.set_ylim([0,ax.get_ylim()[-1]])
ax.tick_params(axis='x', length=5, bottom=True, direction='out', width=0.25)
ax.set_xticklabels(x_dates[::-1].iloc[xt], rotation=60)



ax.annotate(tCredStr, fontsize=8, va='top', ha='left',
              xy=(0.05,0.95), xycoords='axes fraction')
fig.suptitle('Tues Apr 14: Growth gradually slowing in Wisconsin area',
             fontsize=16, fontname='Roboto', fontweight='light',
             x=0.05, y=1.01, ha='left', va='top')





if doSave:
    fig.savefig('./fig-output/midwest-doubling-MH-%s.png'%datestr, facecolor=fig.get_facecolor(),
            dpi=300, bbox_inches='tight', pad_inches=0.5)


```

```python

```
