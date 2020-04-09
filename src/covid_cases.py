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
import pytoolsMH as ptMH

mpl.rc('pdf', fonttype=42) # embed fonts on pdf output 

r_ = np.r_

todayx = 0

def plot_state(df, state, params, ax, is_inset=False, is_cases=True, do_plot=True):
    """
    Params:
        is_cases: True means plot cases, False means plot deaths
    Returns:
        params, df : updated param struct, data frame
    """
    global todayx
    desIx = df.state == state
    if is_cases:
        ys = df.loc[desIx,'positive']
    else: # deaths
        ys = df.loc[desIx,'death']

    dtV = pd.to_datetime(df.loc[desIx,'date'], format='%Y%m%d')
    print(f'Latest data for {state}: {dtV.iloc[0]}')
    xs = (dtV - dtV.iloc[-1]) 
    xs = r_[[x.days for x in xs]] + params.loc[state, 'xoff'] #- todayx

    df.loc[desIx,'day0'] = xs
    params.loc[state,'plot_data'] = [{'xs':xs, 'ys':ys, 'dtV': dtV}]

    if do_plot:
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
        #params.loc[state, 'color'] = ph.get_color()  # this is now set up front
    #todayx = np.max((todayx, np.max(xs)))

    return df, params
    
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


def plot_guide_lines(ax, yoffset_mult=1):
    xs = r_[1,10]
    dtL = [2,3,4]
    for (iD,dt) in enumerate(dtL):
        ys = 2**(xs/dt)
        y2 = ys*10**3.4/(iD+1)  # offset each lines
        y2 = y2*yoffset_mult
        ax.plot(xs, y2, '--', lw=0.5, color='0.6')
        ax.annotate('%d days to double'%dt, xy=(7,y2[1]/2), xycoords='data', fontsize=8, color='0.6')

        
def inset(df, params, ax,  ylim, is_inset=True, is_cases=True): 
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    asp = ax.get_aspect()
    with sns.axes_style('darkgrid'):
        r0 = 1.4
        axins = inset_axes(ax, width=1.3*r0, height=2.2*r0, bbox_to_anchor=(1.2,0.25,0.3,0.6), bbox_transform=ax.transAxes)
    axins.set_facecolor('#EAEAF2')
    
    for state in ['DC', 'MD', 'VA']:
        plot_state(df, state, params, ax=axins, is_inset=True, is_cases=is_cases)
    fixups(axins)
    
    DC = df[df['state'] == 'DC']
    todayx = DC['day0'].iloc[0]
    axins.set_xlim(r_[todayx-2.9,todayx+1.0])
    axins.set_xticks([])
    axins.yaxis.set_visible(False)
    axins.set_yticklabels([])
    axins.xaxis.set_visible(False)
    axins.set_yscale('log')
    axins.set_ylim(ylim)
    return axins
    
def case_anno_inset_double(xs, ax, params):
    # put doubling time on inset axes
    for st in ['DC', 'MD', 'VA']:
        dD = params.loc[st, 'plot_data']
        for tSt in r_[0,1,2]:
            ns = r_[0:2]+tSt
            x0 = np.mean(dD['xs'][ns])
            y0 = np.mean(dD['ys'].iloc[ns])
            slope0 = np.log10(dD['ys'].iloc[ns[0]])-np.log10(dD['ys'].iloc[ns[1]])
            double_time = np.log10(2)/slope0
            pct_rise = (dD['ys'].iloc[ns[0]]/dD['ys'].iloc[ns[1]] * 100) - 100
            #tStr = f'{double_time:.0f}'
            tStr = f'{pct_rise:.0f}'
            if st == 'MD' and tSt == 0:
                #tStr = 'Doubling        \ntime (days): ' + tStr
                tStr = 'Growth             \nper day (%): ' + tStr
            ax.annotate(tStr, xy=(x0,y0), va='bottom', ha='right', color=params.loc[st,'color'],
                          xytext=(-1,2), textcoords='offset points')

    


# class PlotDoubling:

#     def __init__(self, stateList=['DC','VA','MD'], paramsC=None):
#         self.params = paramsC
#         self.stateList = stateList

#         self.doubles = pd.DataFrame(columns=stateList)
#         self.pcts = pd.DataFrame(columns=stateList)
#         for st in self.stateList:
#             self.doubles = self._double_time(st, self.doubles)
#             self.pcts = self._pct_rise(st, self.pcts)

#         self.doubles = find_days(self.doubles)
#         self.pcts = find_days(self.pcts)

#         doubles = doubles.replace([np.inf, -np.inf], np.nan)
#         for st in self.stateList:
#             self.doubles[st].fillna((self.doubles[st].mean()), inplace=True)

            
#     def _double_time(self, st, doubles): 
#         dD = self.params.loc[st, 'plot_data']
#         doublesTemp = []
#         pctsTemp = []
#         for tSt in np.arange(0, len(dD['xs'])-2, 1):
#             ns = r_[0:2]+tSt
#             x0 = np.mean(dD['xs'][ns])
#             y0 = np.mean(dD['ys'].iloc[ns])
#             slope0 = np.log10(dD['ys'].iloc[ns[0]])-np.log10(dD['ys'].iloc[ns[1]])
#             double_time = np.log10(2)/slope0
#             doublesTemp.append(double_time)
#         doubles[st] = doublesTemp
#         return doubles 

    
#     def _pct_rise(self, st, pcts):
#         dD = self.params.loc[st, 'plot_data']
#         pctsTemp = []
#         for tSt in np.arange(0, len(dD['xs'])-2, 1):
#             ns = r_[0:2]+tSt
#             x0 = np.mean(dD['xs'][ns])
#             y0 = np.mean(dD['ys'].iloc[ns])
#             slope0 = np.log10(dD['ys'].iloc[ns[0]])-np.log10(dD['ys'].iloc[ns[1]])
#             pct_rise = (dD['ys'].iloc[ns[0]]/dD['ys'].iloc[ns[1]] * 100) - 100
#             pctsTemp.append(pct_rise)
#         pcts[st] = pctsTemp
#         return pcts

    
#     def find_days(df): 
#         df = df.reindex(index=df.index[::-1])
#         df = df.reset_index(drop = True)
#         df = df.reset_index()
#         df = df.rename(columns = {'index': 'day'})
#         return df
