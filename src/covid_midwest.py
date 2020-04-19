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
from argparse import Namespace
import datetime, dateutil.parser


mpl.rc('pdf', fonttype=42) # embed fonts on pdf output
sns.set_palette('deep')
sns.set_palette('tab10')
#sns.set_palette('default')

r_ = np.r_

todayx = 0

def get_cred_str():
    """from today"""
    tDStr = datetime.date.today().strftime('%b %-d 2020')
    tCredStr = 'Updated %s, 20:00 EDT\n  data: http://covidtracking.com\nGraphic: Hannah Goldbach, Mark Histed\n  @hannah_goldbach @histedlab' % tDStr
    return(tCredStr)

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
    
    for state in ['WI', 'IL']:
        plot_state(df, state, params, ax=axins, is_inset=True, is_cases=is_cases)
    fixups(axins)
    
    WI = df[df['state'] == 'WI']
    todayx = WI['day0'].iloc[0]
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
    for st in ['WI', 'IL']:
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

    


def sub_date_xlabels(params, ax, ylim=None, dtV=None):
    """
    Args:
        params: only used if dtV is not None
        ax:
        ylim:
        dtV:

    Returns:

    """
    if dtV is None:
        dtV = params.loc['WI', 'plot_data']['dtV']

    x_dates = dtV.dt.strftime('%b %-d')
    xt = r_[len(x_dates) - 1:0:-7][::-1]
    ax.set_xticks(xt - 1)
    ax.set_ylim([0, ax.get_ylim()[-1]])
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.tick_params(axis='x', length=5, bottom=True, direction='out', width=0.25)
    ax.set_xticklabels(x_dates[::-1].iloc[xt], rotation=60)


class PlotDoubling:

    def __init__(self, stateList=['WI', 'IL'], params=None, smoothSpan=7):
        """Do all data manip in __init__ - data is in paramsC[plot_data]"""
        
        self.params = params.copy()
        self.stateList = stateList
        self.smoothSpan = smoothSpan
        sns.set_palette('tab10')  # mpl default

        self.doubles = pd.DataFrame(columns={s for s in stateList})  # depends on a dict with no values... ??
        self.pcts = pd.DataFrame(columns={s for s in stateList})
        self.increment = {}

        for st in self.stateList:
            # doubling time
            dD = self.params.loc[st, 'plot_data']
            slope0 = np.diff(np.log10(dD['ys'].to_numpy()))
            double_time = -np.log10(2)/slope0
            self.increment[st] = -np.diff(dD['ys'].to_numpy())
            self.doubles[st] = double_time

            # pct rise
            """pctsTemp = []
            for tSt in np.arange(0, len(dD['xs'])-2, 1):
                ns = r_[0:2]+tSt
                x0 = np.mean(dD['xs'][ns])
                y0 = np.mean(dD['ys'].iloc[ns])
                slope0 = np.log10(dD['ys'].iloc[ns[0]])-np.log10(dD['ys'].iloc[ns[1]])
                pct_rise = (dD['ys'].iloc[ns[0]]/dD['ys'].iloc[ns[1]] * 100) - 100
                pctsTemp.append(pct_rise)
            self.pcts[st] = pctsTemp"""

        # reindex with helper below
        self.doubles = self._find_days(self.doubles)
        self.pcts = self._find_days(self.pcts)

        # do some na/nan adjustment
        self.doubles = self.doubles.replace([np.inf, -np.inf], np.nan)
        self.params['nanIx'] = [[]] * len(self.params)
        for st in self.stateList:
            nanIx = np.isnan(self.doubles[st].to_numpy())
            self.params['nanIx'][st] = [nanIx]
            self.doubles[st].fillna((self.doubles[st].mean()), inplace=True)

        # smoothing
        for st in self.stateList:
            tV =ptMH.math.smooth_lowess(self.doubles[st], x=None, span=smoothSpan, robust=False, iter=None, axis=-1)
            self.doubles[st+'_smooth'] = tV
            
    def _find_days(self, df): 
        df = df.reindex(index=df.index[::-1])
        df = df.reset_index(drop = True)
        df = df.reset_index()
        df = df.rename(columns = {'index': 'day'})
        return df


    def fig_increment(self, doSave=True, title_str='', ylim=None, cred_left=False, yname='cases'):
        """3-panel incremental bar plot"""

        datestr = datetime.datetime.now().strftime('%y%m%d')

        sns.set_style('whitegrid')
        nRows = 3
        fig = plt.figure(figsize=r_[4, 3*nRows]*0.9, dpi=100)
        gs = mpl.gridspec.GridSpec(nRows,1)

        for (iS,st) in enumerate(self.stateList):
            ax = plt.subplot(gs[iS])
            ax.set_facecolor('#fffdfe')

            nanIx = self.params.loc[st, 'nanIx'][0]
            ys0 = self.increment[st][::-1]
            ys0[ys0<0] = 0
            plt.bar(self.doubles['day'], ys0, color=self.params.loc[st, 'color'])
            #ys0[nanIx] = np.nan
            #pH, = plt.plot(self.doubles['day'], ys0, alpha = 0.8, lw = 0.75)
            #ys0 = self.doubles[st+'_smooth']; ys0[nanIx] = np.nan
            #plt.plot(self.doubles['day'], ys0, label = st, lw = 2.5, color = pH.get_color())
            #ast_double = self.doubles[st+'_smooth'].iloc[-1]
            #self.params.loc[st, 'last_double'] = last_double
            #self.params.loc[st, 'color'] = pH.get_color()
        
            plt.grid(False, which='both', axis='x')
            sns.despine(left=True, right=True, top=True, bottom=False)

            sub_date_xlabels(self.params, ax, ylim=ylim) # date labels

            # ax fixups
            if iS == 1:
                ax.set_ylabel('number of %s per day'%yname, fontsize=12)


            ax.annotate(self.params.loc[st, 'fullname'], xy=(0.02,0.98), xycoords='axes fraction',
                        ha='left', va='top', fontsize=14, fontweight='bold')

        plt.tight_layout(h_pad=2)
        
        ap0 = {'ha': 'left', 'xy': (1.2, 0.02) }
        ax.annotate(get_cred_str(), fontsize=8, va='bottom', xycoords='axes fraction', **ap0)

        tStr = datetime.date.today().strftime('%a %B %-d')
        fig.suptitle('%s: %s' % (tStr, title_str),
                     fontsize=16, fontname='Roboto', fontweight='light',
                     x=0.05, y=1.01, ha='left', va='bottom')

        if doSave:
            fig.savefig('./fig-output/midwest-increment-MH-%s-%s.png'%(yname, datestr), facecolor=fig.get_facecolor(),
                    dpi=300, bbox_inches='tight', pad_inches=0.5)
            
    def fig_lowess_cases(self, doSave=True, title_str='', ylim=None, cred_left=False, yname='cases'):
        """3-panel incremental lowess cases plot"""

        datestr = datetime.datetime.now().strftime('%y%m%d')

        sns.set_style('whitegrid')
        nRows = 3
        fig = plt.figure(figsize=r_[4, 3*nRows]*0.9, dpi=100)
        gs = mpl.gridspec.GridSpec(nRows,1)

        for (iS,st) in enumerate(self.stateList):
            ax = plt.subplot(gs[iS])
            ax.set_facecolor('#fffdfe')

            nanIx = self.params.loc[st, 'nanIx'][0]
            ys0 = self.increment[st][::-1]
            ys0[ys0<0] = 0
           
            ys0_smooth = ptMH.math.smooth_lowess(ys0, x=None, span=self.smoothSpan, robust=False, iter=None, axis=-1)
               
            plt.plot(self.doubles['day'], ys0_smooth, color=self.params.loc[st, 'color'])
            plt.axvline(x = 33, color = '0.5', alpha = 0.5)
            plt.axvline(x = 38, color = '0.5', alpha = 0.5)
            plt.axvline(x = 40, color = '0.5', alpha = 0.5)
            plt.annotate('election day', xy = [31, 50], fontsize = 7, rotation = 90)
            plt.annotate('incubation time', xy = [35, 50], fontsize = 7, rotation = 90)
            plt.annotate('lab time', xy = [38.5, 50], fontsize = 7, rotation = 90)
            #ys0[nanIx] = np.nan
            #pH, = plt.plot(self.doubles['day'], ys0, alpha = 0.8, lw = 0.75)
            #ys0 = self.doubles[st+'_smooth']; ys0[nanIx] = np.nan
            #plt.plot(self.doubles['day'], ys0, label = st, lw = 2.5, color = pH.get_color())
            #ast_double = self.doubles[st+'_smooth'].iloc[-1]
            #self.params.loc[st, 'last_double'] = last_double
            #self.params.loc[st, 'color'] = pH.get_color()
        
            plt.grid(False, which='both', axis='x')
            sns.despine(left=True, right=True, top=True, bottom=False)

            sub_date_xlabels(self.params, ax, ylim=ylim) # date labels

            # ax fixups
            if iS == 1:
                ax.set_ylabel('number of %s per day'%yname, fontsize=12)


            ax.annotate(self.params.loc[st, 'fullname'], xy=(0.02,0.98), xycoords='axes fraction',
                        ha='left', va='top', fontsize=14, fontweight='bold')

        plt.tight_layout(h_pad=2)
        
        ap0 = {'ha': 'left', 'xy': (1.2, 0.02) }
        ax.annotate(get_cred_str(), fontsize=8, va='bottom', xycoords='axes fraction', **ap0)

        tStr = datetime.date.today().strftime('%a %B %-d')
        fig.suptitle('%s: %s' % (tStr, title_str),
                     fontsize=16, fontname='Roboto', fontweight='light',
                     x=0.05, y=1.01, ha='left', va='bottom')

        if doSave:
            fig.savefig('./fig-output/midwest-lowess-MH-%s-%s.png'%(yname, datestr), facecolor=fig.get_facecolor(),
                    dpi=300, bbox_inches='tight', pad_inches=0.5)
        
        
    def plot_doubling(self, doSave=True, title_str='', ylim=None, cred_left=False, yname='cases'):
        """Single panel doubling time plot"""
        datestr = datetime.datetime.now().strftime('%y%m%d')
        
        sns.set_style('whitegrid')
        fig = plt.figure(figsize=r_[4, 3]*1.5, dpi=100)
        ax = plt.subplot()
        ax.set_facecolor('#f7fdfe')

        self.params = self.params.astype({'color': 'O'})
        for (iS,st) in enumerate(self.stateList):
            nanIx = self.params.loc[st, 'nanIx'][0]
            ys0 = self.doubles[st]; ys0[nanIx] = np.nan
            pH, = plt.plot(self.doubles['day'], ys0, alpha = 0.8, lw = 0.75, color=self.params.loc[st,'color'])
            ys0 = self.doubles[st+'_smooth']; ys0[nanIx] = np.nan
            plt.plot(self.doubles['day'], ys0, label = st, lw = 2.5, color = pH.get_color())
            last_double = self.doubles[st+'_smooth'].iloc[-1]
            self.params.loc[st, 'last_double'] = last_double
            #self.params.loc[st, 'color'] = pH.get_color()

        # last_double annotate    
        for (iS,st) in enumerate(self.stateList):
            last_double = self.params.loc[st, 'last_double']
            xy = (1.05, 0.65-iS*0.08)
            tStr = f'{st}: {last_double:.2g}'
            if iS == 0:
                tStr = tStr + ' days'
            ax.annotate(tStr, xy=xy, 
                        xycoords='axes fraction', color=self.params.loc[st, 'color'], ha='left',
                        fontweight='bold', fontsize=14)

        plt.ylabel('doubling time for %s (days)'%yname, fontsize=12)

        plt.grid(False, which='both', axis='x')
        sns.despine(left=True, right=True, top=False, bottom=False)

        sub_date_xlabels(self.params, ax, ylim=ylim)# date labels

        if cred_left == True:
            ap0 = {'ha': 'left', 'xy': (0.02,0.01)}
        else:
            ap0 = {'ha': 'right', 'xy': (0.98, 0.01) }
        ax.annotate(get_cred_str(), fontsize=8, va='bottom', xycoords='axes fraction', **ap0)

        tStr = datetime.date.today().strftime('%a %B %-d')
        fig.suptitle('%s: %s' % (tStr, title_str),
                     fontsize=16, fontname='Roboto', fontweight='light',
                     x=0.05, y=1.01, ha='left', va='top')

        ax.annotate('improvement\n(slower growth)', xy=(0.5,0.8), xycoords='figure fraction',
                    textcoords='offset points', xytext=(0,-60),
                    arrowprops=dict(arrowstyle='-|>,head_width=0.4,head_length=0.8', connectionstyle='arc3', color='0.4', lw=1),
                    color='0.3', ha='center')

        if doSave:
            fig.savefig('./fig-output/midwest-doubling-MH-%s-%s.png'%(yname, datestr), facecolor=fig.get_facecolor(),
                    dpi=300, bbox_inches='tight', pad_inches=0.5)




class PlotTesting:

    def __init__(self, ctDf, stateL=['WI', 'IL']):
        """
        Args:
            ctDf:
            stateL:
        Sets:
            self.datD
        """
        self.ctDf = ctDf.copy()

        # iterate through states, compute incremental tests per day, save in self.datD
        self.datD = {}
        self.stateL = stateL
        self.params = pd.DataFrame(index=['WI', 'IL'],
                                   data={'colors': sns.color_palette('deep')[:2],
                                         'fullname': ['Wisconsin', 'Illinois']})

        for state in self.stateL:
            desIx = self.ctDf.state == state
            stDf = self.ctDf.loc[desIx, :].copy()
            stDf.set_index('date', inplace=True, drop=False)

            posV = stDf.loc[:, 'positive'][::-1]
            negV = stDf.loc[:, 'negative'][::-1]
            pdV = np.diff(posV)
            ndV = np.diff(negV)
            pctPos = pdV / (pdV + ndV) * 100
            dtV = pd.to_datetime(stDf['date'], format='%Y%m%d')
            xs = r_[:len(dtV)]

            self.datD[state] = Namespace(posV=posV, negV=negV, pdV=pdV, ndV=ndV, pctPos=pctPos,
                                         dtV=dtV, xs=xs)


    def fig_multipanel_test(self, doSave=False):
        sns.set_style('darkgrid')

        tDStr = datetime.date.today().strftime('%B %-d')
        daylabel = f'Days - last is {tDStr}'

        fig = plt.figure(figsize=r_[1,0.75]*[2,3]*5, dpi=100)
        gs = mpl.gridspec.GridSpec(3,2)

        ax = plt.subplot(gs[0:2,0])
        for state in self.stateL:
            dd = self.datD[state]
            plt.plot(dd.pctPos, '.-', label=state)

        ax.set_title('Percent of tests positive, WI area')
        ax.set_ylabel('Positive tests per day (%)')
        ax.set_xlabel(daylabel)
        # markup line
        maxN = len(self.datD['WI'].pctPos)
        meanNDay = 7
        desNs = r_[maxN-meanNDay:maxN]
        tV = np.hstack([self.datD[x].pctPos[desNs] for x in self.datD.keys()])
        tM = np.mean(tV)
        plt.plot(desNs, tM+desNs*0, color='k', lw=5, ls='-', alpha=0.5)
        # anno it
        ax.annotate('mean\npos test rates,\nlast %d days'%meanNDay,
                    xy=(desNs[2],tM), xycoords='data', xytext=(-50,+50), textcoords='offset points',
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.1', color='k'),
                    color='0.3', ha='center')
        ax.annotate('no neg tests\nreported by MD here',
                    xy=(16,100), xycoords='data', xytext=(-10,-60), textcoords='offset points',
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3', color='0.3'),
                    color='0.3', ha='center')

        # next three panels
        ax = plt.subplot(gs[0,1])
        ax2 = plt.subplot(gs[1,1])
        ax3 = plt.subplot(gs[2,1])
        for state in ['WI', 'IL']:
            dd = self.datD[state]
            display(len(dd.posV))
            pH = ax.plot(dd.pdV, '.-', label=state)
            pH = ax2.plot(dd.ndV, '.-', label=state)
            pH = ax3.plot(dd.ndV+dd.pdV, '.-', label=state)
        ax.legend()
        ax.set_title('Positive results per day')
        ax.set_ylabel('Results')

        ax2.set_title('Negative results per day')
        ax2.set_ylabel('Results')

        ax3.set_title('Total results per day')
        ax3.set_ylabel('Results')
        ax3.set_xlabel(daylabel)

        # suptitle and extra fig strings
        fig.suptitle(f"{datetime.date.today().strftime('%a %B %-d')}: \n"
                     'Test rates, WI area (WI, IL)',
                     fontsize=16, fontname='Roboto', fontweight='light',
                     x=0.05, y=0.92, ha='left', va='bottom')

        if doSave:
            fig.savefig('./fig-output/midwest-testing-%s.png'%datestr,
                    dpi=300, bbox_inches='tight', pad_inches=0.5)
                    #bbox_inches=r_[0,0,10,15])#,
        return fig


    def fig_pos_test_rate(self, ylim=None, title_str='', nbootreps=100, doSave=False):
        sns.set_style('whitegrid')
        fig = plt.figure(figsize=r_[3.8, 3]*r_[2,3]*1, dpi=100)
        gs = mpl.gridspec.GridSpec(3,2)

        for (iS,st) in enumerate(self.stateL):
            ax = plt.subplot(gs[iS,0])
            ax.set_facecolor('#f8fafb')#'#f7fdfe')
            dD = self.datD[st]
            ys0 = dD.pctPos
            xs0 = dD.xs[1:]-1 # drop the first element because plotting against a difference
            pH, = plt.plot(xs0, ys0, alpha = 0.8, lw = 0.75, color=self.params.loc[st,'colors'])
            desIx = (xs0 > 15) & (ys0 < 100)
            smD = { 'span': 9 }
            ysSm = ptMH.math.smooth_lowess(ys0[desIx], xs0[desIx], **smD)
            (ylow, yhigh, out_xs, boot_mat) = ptMH.math.smooth_lowess_bootstrap(ys0[desIx], xs0[desIx],
                                                                                nbootreps=nbootreps, **smD)

            pH2, = ax.plot(xs0[desIx], ysSm, lw=5, color=pH.get_color())
            ax.fill_between(out_xs, ylow, yhigh, alpha=0.2, color=pH.get_color(), edgecolor=None, lw=0)

            sub_date_xlabels(params=None, ax=ax, ylim=ylim, dtV=self.datD['WI'].dtV)  # date labels
            #if iS != 2:
            #    ax.set_xticklabels('')
            ax.set_ylim(0,59)
            ax.set_xlim((6,len(xs0)))


            plt.grid(False, which='both', axis='x')
            sns.despine(left=True, right=True, top=True, bottom=False)
            ax.annotate(self.params.loc[st, 'fullname'], xy=(0.04,0.95), xycoords='axes fraction',
                        ha='left', va='top', fontsize=14, fontweight='bold')


            plt.ylabel('pos. test rates (%)', fontsize=12)



        plt.tight_layout(h_pad=2, pad=1) #rect=(0,0,0,0.95))

        # names annotate
        ap0 = {'ha': 'left', 'xy': (1.2, 0.02)}
        ax.annotate(get_cred_str(), fontsize=8, va='bottom', xycoords='axes fraction', **ap0)
        # 95% CI text
        ax0 = plt.subplot(gs[0,0])
        ax0.annotate('95% CI', fontsize=11, va='top', ha='left', fontweight='medium',
                     color=self.params.loc['WI','colors'], alpha=0.5,
                     xycoords='data', xy=(20,15))
        # guide arrow
        ax0.annotate('improvement\n(tests less likely\n to be positive)', xy=(1.2,0.5), xycoords='axes fraction',
                    textcoords='offset points', xytext=(0,40),
                    arrowprops=dict(arrowstyle='-|>,head_width=0.4,head_length=0.8', connectionstyle='arc3', color='0.4', lw=1),
                    color='0.3', ha='center')
        # suptitle
        tStr = datetime.date.today().strftime('%a %B %-d')
        fig.suptitle('%s: %s' % (tStr, title_str),
                     fontsize=16, fontname='Roboto', fontweight='light',
                     x=0, y=1.01, ha='left', va='bottom')

        datestr = datetime.datetime.now().strftime('%y%m%d')
        if doSave:
            fig.savefig('./fig-output/midwest-posrate-MH-%s.png'%datestr, facecolor=fig.get_facecolor(),
                    dpi=300, bbox_inches='tight', pad_inches=0.5)