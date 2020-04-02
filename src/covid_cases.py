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

mpl.rc('pdf', fonttype=42) # embed fonts on pdf output 

r_ = np.r_

def plot_state(df, state, params, ax, is_inset):
    global todayx
    todayx = 0
    desIx = df.state == state
    ys = df.loc[desIx,'positive']
    dtV = pd.to_datetime(df.loc[desIx,'date'], format='%Y%m%d')
    print(f'Latest data for {state}: {dtV.iloc[0]}')
    xs = (dtV - dtV.iloc[-1]) 
    xs = r_[[x.days for x in xs]] + params.loc[state, 'xoff'] #- todayx

    df.loc[desIx,'day0'] = xs


    ph, = ax.plot(xs, ys, marker='.', label=state, lw=2, markersize=9)
    if state in params.index:
        if is_inset:
            xytext = r_[7,0]
            xy=(xs[0],ys.iloc[0])
        else:
            xytext = (params.loc[state,'labXOff'], params.loc[state,'labYOff'])
            xy=(xs[1],ys.iloc[1])
            
        ah = ax.annotate(state, 
                         xy=xy, xycoords='data', xytext=xytext, textcoords='offset points',
                         color=ph.get_color(),
                         fontweight='bold', fontsize=12)

        lw = params.loc[state,'lw']
        ph.set_linewidth(lw)
        if lw < 1:
            ph.set_markersize(3)
            ph.set_color('0.4')
            ah.set_color('0.4')
    todayx = np.max((todayx, np.max(xs)))
    
def fixups(ax):
    lp = r_[1,2,5]
    yt = np.hstack((lp*10,lp*10**2,lp*10**3,lp*10**4,10**5))
    #ax.set_yticks(yt)
    ax.set_yscale('log')
    plt.setp(ax.get_xticklabels(), fontsize=9)
    plt.setp(ax.get_yticklabels(), fontsize=9)
    ax.yaxis.set_major_locator(mpl.ticker.FixedLocator(yt, nbins=len(yt)+1))
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x,pos: '{:,.0f}'.format(x)))
    ax.set_ylabel('Cases', fontsize=13)
    ax.set_xlabel('Days', fontsize=13)
        
def inset(df, params, ax,  ylim, is_inset=True): 
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    asp = ax.get_aspect()
    with sns.axes_style('darkgrid'):
        r0 = 1.4
        axins = inset_axes(ax, width=1.3*r0, height=2.2*r0, bbox_to_anchor=(1.2,0.25,0.3,0.6), bbox_transform=ax.transAxes)
    axins.set_facecolor('#EAEAF2')
    
    for state in ['DC', 'MD', 'VA']:
        plot_state(df, state, params, ax=axins, is_inset=True)
    fixups(axins)
    
    DC = df[df['state'] == 'DC']
    todayx = DC['day0'].iloc[0]
    axins.set_xlim([todayx-2.9,todayx+1.0])
    axins.set_xticks([])
    axins.yaxis.set_visible(False)
    axins.set_yticklabels([])
    axins.xaxis.set_visible(False)
    axins.set_yscale('log')
    axins.set_ylim(ylim*1.2)
    
def case_double(xs, dtL, ax):
    for (iD,dt) in enumerate(dtL):
        ys = 2**(xs/dt)
        y2 = ys*10**3.4/(iD+1)
        ax.plot(xs, y2, '--', lw=0.5, color='0.6')
        ax.annotate('%d days to double'%dt, xy=(7,y2[1]/2), xycoords='data', fontsize=8, color='0.6')
    