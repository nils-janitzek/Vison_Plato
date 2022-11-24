#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""

vison pipeline: Classes to do plots.

Created on Mon Nov 13 17:54:08 2017

:author: Ruyman Azzollini

"""
# IMPORT STUFF
from pdb import set_trace as stop
import numpy as np
from collections import OrderedDict
from vison.datamodel.HKtools import format_date
import copy
import itertools
import os
import string as st
import gc

import matplotlib
# matplotlib.use('Agg')
from matplotlib import pyplot as plt
matplotlib.rcParams['font.size'] = 17
matplotlib.rc('xtick', labelsize=14)
matplotlib.rc('axes', linewidth=1.1)
matplotlib.rcParams['legend.fontsize'] = 12
matplotlib.rcParams['legend.handlelength'] = 3
matplotlib.rcParams['xtick.major.size'] = 5
matplotlib.rcParams['ytick.major.size'] = 5
matplotlib.rcParams['image.interpolation'] = 'none'
import matplotlib.cm as cm
from matplotlib import colors as mcolors
from mpl_toolkits.mplot3d import Axes3D
# END IMPORT


def _squiggle_xy(a, b, c, d, i=np.arange(0.0, 2 * np.pi, 0.05)):
    """Dummy function used for Tests only."""
    return np.sin(i * a) * np.cos(i * b), np.sin(i * c) * np.cos(i * d)


class BasicPlot(object):

    def __init__(self, **kwargs):

        self.fig = None
        self.figsize = (9, 9)
        if 'fig' in kwargs:
            self.fig = kwargs['fig']
        self.axs = []

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        gc.collect()
        return isinstance(value, TypeError)

    def populate_axes(self):
        raise NotImplementedError("Subclass must implement abstract method")

    def plt_trimmer(self):
        raise NotImplementedError("Subclass must implement abstract method")

    def init_fig(self):
        plt.close('all')
        self.fig = plt.figure(figsize=self.figsize)

    def render(self, figname=''):

        self.init_fig()

        self.populate_axes()

        self.plt_trimmer()

        # figname = '' # TESTS

        if figname == '':
            plt.show()
        else:
            plt.savefig(figname)
            #import pickle as pl
            #pl.dump(self.fig, file('test.pickle','w'))

        plt.close(self.fig)
        plt.close('all')
        plt.clf()
        gc.collect()


class ShellPlot(BasicPlot):
    """ """

    def __init__(self, data, **kwargs):

        self.data = data
        self.plotter = kwargs['plotter']
        kwargs.pop('plotter')
        self.meta = dict()
        self.meta.update(kwargs)

    def render(self, figname=''):
        kwargs = self.meta.copy()
        kwargs['figname'] = figname
        self.plotter(self.data, **kwargs)


class XYPlot(BasicPlot):

    def __init__(self, data, **kwargs):

        super(XYPlot, self).__init__(**kwargs)

        meta = dict(suptitle='',
                    doLegend=False,
                    doNiceXDate=False,
                    doYErrbars=False,
                    doConfidence=False)

        meta.update(kwargs)

        self.figsize = (8, 7)
        self.data = copy.deepcopy(data)
        self.meta = dict()
        self.meta.update(meta)

        self.handles = []
        self.labels = []

        self.corekwargs = dict()
        if 'corekwargs' in kwargs:
            self.corekwargs.update(kwargs['corekwargs'])

    def _ax_core_funct(self, key=''):
        """ """

        ckwargs = self.corekwargs.copy()

        if key != '':
            xarr = self.data['x'][key]
            yarr = self.data['y'][key]

            label = key.replace('_', '\_')
            kwargs = dict(label=label, marker='.', linestyle='')
            if key in ckwargs:
                kwargs.update(ckwargs[key])
            else:
                kwargs.update(ckwargs)
            handle = self.ax.plot(xarr, yarr, **kwargs)

            if len(handle)>1:
                handle = [handle[-1]]

            if self.meta['doYErrbars']:
                eyarr = self.data['ey'][key]
                self.ax.errorbar(xarr, yarr, yerr=eyarr, color='k', fmt='', linestyle='')
        else:
            xarr = self.data['x']
            yarr = self.data['y']
            kwargs = dict(marker='.', linestyle='')
            kwargs.update(ckwargs)
            self.ax.plot(xarr, yarr, **kwargs)
            handle, label = None, None
            if self.meta['doYErrbars']:
                eyarr = self.data['ey']
                self.ax.errorbar(xarr, yarr, yerr=eyarr, 
                    color='k', fmt='', linestyle='')

        return handle, label

    def plot_confidence_intervals(self):
        """ """
        condict = self.data['confidence'].copy()
        x = condict['x'].copy()
        yminus = condict['yminus'].copy()
        yplus = condict['yplus'].copy()

        conkwargs = self.meta['confidence_kwargs'].copy()

        self.ax.fill_between(x, yminus, yplus, **conkwargs)


    def populate_axes(self):

        self.ax = self.fig.add_subplot(111)

        try:
            labelkeys = self.data['labelkeys']
        except KeyError:
            labelkeys = []

        if len(labelkeys) > 0:

            for labelkey in labelkeys:
                handle, label = self._ax_core_funct(labelkey)
                self.handles += handle
                self.labels.append(label)

        else:

            _, _ = self._ax_core_funct()

        if self.meta['doNiceXDate']:
            _xticks = self.ax.get_xticks()
            if len(_xticks) > 6:
                self.ax.set_xticks(_xticks[::2])

        if self.meta['doConfidence']:
            self.plot_confidence_intervals()

        if 'title' in self.meta:
            self.ax.set_title(self.meta['title'])

        if 'xlabel' in self.meta:
            self.ax.set_xlabel(self.meta['xlabel'])
        if 'ylabel' in self.meta:
            self.ax.set_ylabel(self.meta['ylabel'])

        if 'ylim' in self.meta:
            self.ax.set_ylim(self.meta['ylim'])
        if 'xlim' in self.meta:
            self.ax.set_xlim(self.meta['xlim'])

        if 'text' in self.meta:
            textdict = dict(kwargs=dict())
            textdict.update(self.meta['text'])
            self.ax.text(textdict['x'],textdict['y'],textdict['text'],
                transform=self.ax.transAxes,
                **textdict['kwargs'])

    def plt_trimmer(self):

        if self.meta['doLegend']:

            plt.figlegend(self.handles, self.labels, loc='center right')

        if self.meta['doNiceXDate']:

            plt.gca().xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(format_date))
            self.fig.autofmt_xdate()

        if self.meta['suptitle'] != '':
            plt.subplots_adjust(top=0.95)
            plt.suptitle(self.meta['suptitle'])

        plt.tight_layout()

        if self.meta['doLegend']:
            plt.subplots_adjust(right=0.85)


class HistoPlot(XYPlot):

    def __init__(self, data, **kwargs):

        super(HistoPlot, self).__init__(data, **kwargs)

    def _ax_core_funct(self, key=''):
        """ """

        ckwargs = self.corekwargs.copy()

        if key != '':
            bins = self.data['x'][key]
            h = self.data['y'][key]

            label = key.replace('_', '\_')
            kwargs = dict(label=label, weights=None, cumulative=False,
                    histtype='step', align='mid', orientation='vertical', log=False)
            if key in ckwargs:
                kwargs.update(ckwargs[key])
            else:
                kwargs.update(ckwargs)
            #kwargs = dict()
            _, _, handle = self.ax.hist(h, bins=bins, **kwargs)
            #handle = self.ax.plot(bins,h)

            #if len(handle)>1:
            #    handle = [handle[-1]]

            if self.meta['doYErrbars']:
                eyarr = self.data['ey'][key]
                self.ax.errorbar(bins, h, yerr=eyarr, color='k', fmt='', linestyle='')
        else:
            bins = self.data['x']
            h = self.data['y']
            kwargs = dict(weights=None, cumulative=False,
                    histtype='step', align='mid', orientation='vertical', log=False)
            kwargs.update(ckwargs)

            self.ax.hist(h, bins=bins, **kwargs)

            handle, label = None, None
            if self.meta['doYErrbars']:
                eyarr = self.data['ey']
                self.ax.errorbar(bins, h, yerr=eyarr, color='k', fmt='', linestyle='')

        return handle, label


class CCD2DPlot(BasicPlot):

    def __init__(self, data, **kwargs):

        super(CCD2DPlot, self).__init__(**kwargs)

        meta = dict(suptitle='',
                    doLegend=False,
                    doColorbar=False,
                    doNiceXDate=False,
                    doRotateXLabels=False)

        meta.update(kwargs)

        self.figsize = (8, 8)
        self.Quads = ['F', 'E']
        self.data = copy.deepcopy(data)
        self.meta = dict()
        self.meta.update(meta)
        self.handles = []
        self.labels = []
        self.fig = None
        self.axs = dict()
        self.axarr = []

        self.corekwargs = dict()
        if 'corekwargs' in kwargs:
            self.corekwargs.update(kwargs['corekwargs'])

    def init_fig(self):
        self._init_fig_and_axes()

    def _init_fig_and_axes(self):
        """ """
        plt.close('all')
        fig, axsarr = plt.subplots(
            2, 2, sharex=False, sharey=True, figsize=self.figsize)
        self.fig = fig

        self.axsarr = axsarr

        # initialisation of self.axs

        for Q in self.Quads:
            self.axs[Q] = None

        self.axs['F'] = self.axsarr[0, 0]

        plotlist = ['F', 'E']

        for k in range(len(plotlist)):
            Q = plotlist[k]
            self.axs[Q] = self.axsarr.flatten()[k]

    def _ax_core_funct(self, ax, Qdict, key=''):
        raise NotImplementedError("Subclass must implement abstract method")

    def populate_axes(self):

        try:
            labelkeys = self.data['labelkeys']
        except KeyError:
            labelkeys = []

        for iQ, Q in enumerate(self.Quads):

            ax = self.axs[Q]
            Qdict = self.data[Q]

            if len(labelkeys) > 0:
                for labelkey in labelkeys:
                    handle, label = self._ax_core_funct(ax, Qdict, labelkey)
                    if Q == 'E':
                        self.handles += handle
                        self.labels.append(label)
            else:
                _, _ = self._ax_core_funct(ax, Qdict)

            if Q in ['F']:
                ax.text(0.05, 0.9, Q, horizontalalignment='left',
                        transform=self.axs[Q].transAxes)
            elif Q in ['E']:
                ax.text(0.9, 0.9, Q, horizontalalignment='right',
                        transform=self.axs[Q].transAxes)

            if self.meta['doNiceXDate']:
                _xticks = ax.get_xticks()
                if len(_xticks) > 6:
                    ax.set_xticks(_xticks[::2])

            if 'xlabel' in self.meta and Q in ['F']:
                ax.set_xlabel(self.meta['xlabel'])
            if 'ylabel' in self.meta and Q in ['E']:
                ax.set_ylabel(self.meta['ylabel'])

            if 'ylim' in self.meta:
                ax.set_ylim(self.meta['ylim'])
            if 'xlim' in self.meta:
                ax.set_xlim(self.meta['xlim'])

    def plt_trimmer(self):

        for Q in ['F']:
            plt.setp(self.axs[Q].get_xticklabels(), visible=False)

        for Q in ['E']:
            plt.setp(self.axs[Q].get_yticklabels(), visible=False)

        if self.meta['doRotateXLabels']:
            for Q in self.Quads:
                for tick in self.axs[Q].get_xticklabels():
                    tick.set_rotation(45)

        if self.meta['doLegend']:
            plt.figlegend(self.handles, self.labels, loc='center right')

        if self.meta['doNiceXDate']:
            plt.gca().xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(format_date))
            self.fig.autofmt_xdate()
            # plt.locator_params(nticks=4,axis='x',prune='both')

        plt.locator_params(axis='y', nbins=5, prune='both')

        # plt.locator_params(axis='y',prune='both')
        if not self.meta['doNiceXDate']:
            try:
                plt.locator_params(axis='x', nbins=4, prune='both')
            except BaseException:
                pass

        plt.subplots_adjust(hspace=0.0)
        plt.subplots_adjust(wspace=0.0)

        plt.margins(0.05)

        plt.suptitle(self.meta['suptitle'])
        # plt.tight_layout()
        plt.subplots_adjust(top=0.85)
        if self.meta['doLegend']:
            plt.subplots_adjust(right=0.85)

        if self.meta['doColorbar']:
            #cbar_ax = self.fig.add_axes([0.85, 0.15, 0.05, 0.7])
            #plt.colorbar(cax=cbar_ax, mappable=self.mappables[0],orientation='vertical')

            if ('vmin' in self.corekwargs) and ('vmax' in self.corekwargs):
                vmin = self.corekwargs['vmin']
                vmax = self.corekwargs['vmax']
            else:
                vmin = np.min([np.nanmin(item.get_array()) for item in self.mappables])
                vmax = np.max([np.nanmax(item.get_array()) for item in self.mappables])
            
            norm = mcolors.Normalize(vmin=vmin,vmax=vmax,clip=False)
            scmap = cm.ScalarMappable(norm=norm, cmap=self.mappables[0].cmap)

            self.fig.colorbar(scmap, ax=self.axsarr.flatten().tolist(),
                              orientation='vertical', fraction=.1)


class CCD2DPlotYvsX(CCD2DPlot):

    def _ax_core_funct(self, ax, Qdict, key=''):

        ckwargs = self.corekwargs.copy()

        if key != '':

            xarr = Qdict['x'][key]
            yarr = Qdict['y'][key]

            label = key.replace( '_', '\_')
            kwargs = dict(label=label, marker='.', linestyle='')
            if key in ckwargs:
                kwargs.update(ckwargs[key])
            else:
                kwargs.update(ckwargs)
            handle = ax.plot(xarr, yarr, **kwargs)
        else:
            xarr = Qdict['x']
            yarr = Qdict['y']
            kwargs = dict(marker='.', linestyle='')
            kwargs.update(ckwargs)
            ax.plot(xarr, yarr, **kwargs)
            handle, label = None, None

        return handle, label


class BeamPlot(BasicPlot):
    """ """

    def __init__(self, data, **kwargs):

        super(BeamPlot, self).__init__(**kwargs)
        #print("figclasses.Fig_Beam1DHist data",data)
        #print("\n")

        meta = dict(suptitle='',
                    #ccdtitles=dict(CCD1='CCD1', CCD2='CCD2', CCD3='CCD3'),
                    ccdtitles=dict(CCD1='CCD1', CCD2='CCD2', CCD3='CCD3',CCD4='CCD4'),
                    doLegend=False,
                    doColorbar=False,
                    doNiceXDate=False,
                    doRotateXLabels=False,
                    doYErrbars=False)
        meta.update(kwargs)
        
        self.figsize = (15, 6)
        self.Quads = ['E', 'F']
        #self.CCDs = [1, 2, 3]
        self.CCDs = [1, 2, 3, 4]
        self.data = copy.deepcopy(data)
        self.meta = dict()
        self.meta.update(meta)
        self.handles = []
        self.labels = []
        self.fig = None
        self.axs = dict()
        self.axsarr = []

        self.corekwargs = dict()
        if 'corekwargs' in kwargs:
            self.corekwargs.update(kwargs['corekwargs'])

    def init_fig(self):
        self._init_fig_and_axes()

    def _init_fig_and_axes(self):
        """ """
        plt.close('all')
        fig, axsarr = plt.subplots(
            2, 4, sharex=False, sharey=True, figsize=self.figsize)
        self.fig = fig

        self.axsarr = axsarr

        # initialisation of self.axs

        for CCD in self.CCDs:
            CCDkey = 'CCD%i' % CCD
            self.axs[CCDkey] = dict()
            for Q in self.Quads:
                self.axs[CCDkey][Q] = None

        self.axs['CCD1']['E'] = self.axsarr[0, 0]

        plotlist = [item for item in itertools.product(self.CCDs, ['E'])] +\
                   [item for item in itertools.product(self.CCDs, ['F'])]
        print("plotlist:",plotlist)

        for k in range(1, len(plotlist) + 1):
            CCDkey = 'CCD%i' % plotlist[k - 1][0]
            Q = plotlist[k - 1][1]
            self.axs[CCDkey][Q] = self.axsarr.flatten()[k - 1]
            
    def _ax_core_funct(self, ax, CQdict, key=''):
        """ """
        raise NotImplementedError("Subclass must implement abstract method")

    def populate_axes(self):
        """ """
        #continue here!

        try:
            labelkeys = self.data['labelkeys']
        except KeyError:
            labelkeys = []

        for CCD in self.CCDs:
            CCDkey = 'CCD%i' % CCD
            for Q in self.Quads:

                ax = self.axs[CCDkey][Q]
                print("self.axs",self.axs,ax)
                CQdict = self.data[CCDkey][Q]

                if len(labelkeys) > 0:
                    for labelkey in labelkeys:
                        print("CCD,Q,labelkey:",CCD,Q,labelkey)
                        handle, label = self._ax_core_funct(ax, CQdict, labelkey)
                        if Q == 'E' and CCD == 1:
                            self.handles += handle
                            self.labels.append(label)
                else:
                    _, _ = self._ax_core_funct(ax, CQdict)

                if Q in ['F']:
                    ax.text(0.05, 0.9, Q, horizontalalignment='left',
                            transform=self.axs[CCDkey][Q].transAxes)
                elif Q in ['E']:
                    ax.text(0.9, 0.9, Q, horizontalalignment='right',
                            transform=self.axs[CCDkey][Q].transAxes)

                if Q == 'E':
                    #ax.set_title(CCDkey, x=0.5)
                    ax.set_title(CCDkey)

                if self.meta['doNiceXDate']:
                    _xticks = ax.get_xticks()
                    if len(_xticks) > 6:
                        ax.set_xticks(_xticks[::2])

                if 'xlabel' in self.meta and Q in ['F']:
                    ax.set_xlabel(self.meta['xlabel'])
                #if 'ylabel' in self.meta and Q in ['E'] and CCD == 1:
                #    ax.set_ylabel(self.meta['ylabel'])
                if 'ylabel' in self.meta and CCD == 1:
                    ax.set_ylabel(self.meta['ylabel'])

                #if 'ylim' in self.meta:
                #    ax.set_ylim(self.meta['ylim'])
                #print("Plot:self.data", self.data)
                print("self.meta['suptitle']:",self.meta['suptitle'])
                BIAS_profile_figs=["BIAS0X: Profiles across columns","BIAS0X: Profiles across rows","BIAS0X: STDDEV Profiles across rows"]
                if self.meta['suptitle'] in BIAS_profile_figs:

                    if Q in ['E']:
                        ydata_E=self.data[CCDkey]['E']['y'][labelkey]
                        #print("Plot: ydata_E",ydata_E,np.amin(ydata_E),np.amax(ydata_E))
                        ax.set_ylim(0.9*np.amin(ydata_E),1.1*np.amax(ydata_E))    
                    elif Q in ['F']:
                        ydata_F=self.data[CCDkey]['F']['y'][labelkey]
                        #print("Plot: ydata_F",ydata_F,np.amin(ydata_F),np.amax(ydata_F))
                        ax.set_ylim(0.9*np.amin(ydata_F),1.1*np.amax(ydata_F))    
                    
                if 'xlim' in self.meta:
                    ax.set_xlim(self.meta['xlim'])
                if self.meta['suptitle']=="BIAS0X: RON Distribution":
                    ax.set_xlim(0,6)



        # self.axs[CCDkey][Q].locator_params(nticks=4,axis='x')

    def plt_trimmer(self):

        for CCD in self.CCDs:
            pass
            #for Q in ['F']:
            #    plt.setp(self.axs['CCD%i' % CCD]
            #             [Q].get_xticklabels(), visible=False)
            #if CCD != 1:
            #    for Q in self.Quads:
            #        plt.setp(self.axs['CCD%i' % CCD]
            #                 [Q].get_yticklabels(), visible=False)
            #else:
            #    for Q in ['E']:
            #        plt.setp(self.axs['CCD%i' % CCD]
            #                 [Q].get_yticklabels(), visible=False)
        if self.meta['doRotateXLabels']:
            for CCD in self.CCDs:
                for Q in self.Quads:
                    for tick in self.axs['CCD%i' % CCD][Q].get_xticklabels():
                        tick.set_rotation(45)

        if self.meta['doLegend']:
            plt.figlegend(self.handles, self.labels, loc='center right')

        if self.meta['doNiceXDate']:
            plt.gca().xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(format_date))
            self.fig.autofmt_xdate()
            # plt.locator_params(nticks=4,axis='x',prune='both')

        plt.locator_params(axis='y', nbins=5, prune='both')

        # plt.locator_params(axis='y',prune='both')
        if not self.meta['doNiceXDate']:
            try:
                plt.locator_params(axis='x', nbins=4, prune='both')
            except BaseException:
                pass

        plt.subplots_adjust(hspace=0.0)
        plt.subplots_adjust(wspace=0.0)

        plt.margins(0.05)

        plt.suptitle(self.meta['suptitle'])
        # plt.tight_layout()
        plt.subplots_adjust(top=0.85)
        if self.meta['doLegend']:
            plt.subplots_adjust(right=0.85)

        if self.meta['doColorbar']:
            #cbar_ax = self.fig.add_axes([0.85, 0.15, 0.05, 0.7])
            #plt.colorbar(cax=cbar_ax, mappable=self.mappables[0],orientation='vertical')
            if ('vmin' in self.corekwargs) and ('vmax' in self.corekwargs):
                vmin = self.corekwargs['vmin']
                vmax = self.corekwargs['vmax']
            else:
                vmin = np.min([np.nanmin(item.get_array()) for item in self.mappables])
                vmax = np.max([np.nanmax(item.get_array()) for item in self.mappables])
            
            norm = mcolors.Normalize(vmin=vmin,vmax=vmax,clip=False)
            scmap = cm.ScalarMappable(norm=norm, cmap=self.mappables[0].cmap)

            self.fig.colorbar(scmap, ax=self.axsarr.flatten().tolist(),
                              orientation='vertical', fraction=.1)
            #self.fig.colorbar(self.mappables[0], ax=self.axsarr.flatten().tolist(),
            #                  orientation='vertical', fraction=.1)


class BeamPlotYvX(BeamPlot):

    def _ax_core_funct(self, ax, CQdict, key=''):
        #print("key,CQdict",key,CQdict)
        ckwargs = self.corekwargs.copy()

        if key != '':

            xarr = CQdict['x'][key]
            yarr = CQdict['y'][key]
            
            label = key.replace('_', '\_')
            kwargs = dict(label=label, marker='.', linestyle='')
            if key in ckwargs:
                kwargs.update(ckwargs[key])
            else:
                kwargs.update(ckwargs)
            handle = ax.plot(xarr, yarr, **kwargs)

            if self.meta['doYErrbars']:
                eyarr = CQdict['ey'][key]
                ax.errorbar(xarr, yarr, yerr=eyarr, color='k', fmt='', linestyle='')

        else:
            xarr = CQdict['x']
            yarr = CQdict['y']
            kwargs = dict(marker='.', linestyle='')
            kwargs.update(ckwargs)
            ax.plot(xarr, yarr, **kwargs)
            handle, label = None, None
            if self.meta['doYErrbars']:
                eyarr = CQdict['ey']
                ax.errorbar(xarr, yarr, yerr=eyarr, 
                    color='k', fmt='', linestyle='')
        
        print("Plot _ax_core_funct: len(xarr),len(yarr)", len(xarr),len(yarr))
        if len(xarr)==2295: 
            prescan=25
            image=2255
            overscan=15
            ax.plot([prescan,prescan],[min(yarr),max(yarr)],color="orange",linestyle="--") 
            ax.plot([prescan+image,prescan+image],[min(yarr),max(yarr)],color="green",linestyle="--")

        elif len(xarr)==4539:
            image=4509
            voverscan=30
            #ax.plot([image,image],[min(yarr),max(yarr)],color="orange",linestyle="--")#?
            ax.plot([voverscan,voverscan],[min(yarr),max(yarr)],color="green",linestyle="--")#?
        else:
            print("invalid Naxis dimensions for Plato CCDs")

        return handle, label


class BeamImgShow(BeamPlot):

    mappables = []

    def _ax_core_funct(self, ax, CQdict):

        internals = dict(origin='lower left')
        ckwargs = self.corekwargs.copy()
        internals.update(ckwargs)
        self.mappables.append(ax.imshow(CQdict['img'], **internals))
        handle, label = None, None
        return handle, label


class Beam1DHist(BeamPlot):
    """ """

    def _ax_core_funct(self, ax, CQdict, key=''):

        hist_kwargs = dict(weights=None,
                           cumulative=False, histtype='step', align='mid',
                           orientation='vertical', log=False)

        hist_kwargs.update(self.corekwargs)

        # for mkey in hist_kwargs.keys():
        #    if mkey in self.meta:
        #        hist_kwargs[mkey] = self.meta[mkey]

        if key != '':
            label = key.replace( '_', '\_')
            bins = CQdict['x'][key]
            h = CQdict['y'][key]
            hist_kwargs['label'] = label
            #print bins.min(), bins.max()
        else:
            bins = CQdict['x']
            h = CQdict['y']
            label = None

        #h=1.0#TEST_NJ
        bins=np.arange(0,6.1,0.1)#TEST_NJ
        _, _, patch = ax.hist(h, bins=bins, **hist_kwargs)
        #htest=ax.hist(h, bins=bins, **hist_kwargs)

        print("Beam1DHist_plotting: key,CQdict['x'],CQdict['y'],bins,h:",key,bins,h)
        print("RON h:",h)
        return patch, label


class ImgShow(BasicPlot):

    mappables = []

    def __init__(self, data, **kwargs):

        super(ImgShow, self).__init__(**kwargs)

        defaults = dict(title='',
            doColorbar=False)

        self.figsize = (7, 7)
        self.data = data
        self.meta = dict()
        self.meta.update(defaults)
        self.meta.update(kwargs)

        self.corekwargs = dict()
        if 'corekwargs' in kwargs:
            self.corekwargs.update(kwargs['corekwargs'])
        

    def populate_axes(self):
        """ """
        #internals = dict(origin='lower left')
        internals = dict(origin='lower')#workaround_NJ
        ckwargs = self.corekwargs.copy()
        internals.update(ckwargs)
        self.axs = [self.fig.add_subplot(111)]
        self.mappables.append(self.axs[0].imshow(self.data, **internals))
        self.axs[0].set_title(self.meta['title'])
        print("plot title,asx",self.meta['title'],self.axs)

    
    def plt_trimmer(self):

        if self.meta['doColorbar']:
            #cbar_ax = self.fig.add_axes([0.85, 0.15, 0.05, 0.7])
            #plt.colorbar(cax=cbar_ax, mappable=self.mappables[0],orientation='vertical')
            self.fig.colorbar(self.mappables[0], 
                ax=self.axs[0],
                orientation='vertical', fraction=.1)


def testBeam2DPlot():

    ccddict = dict()
    for iQ, Q in enumerate(['F', 'E']):
        _x, _y = _squiggle_xy(iQ + 1, iQ + 1, iQ + 2, iQ + 2)
        xdict = OrderedDict(foo=_x, bar=_x * 2.)
        ydict = OrderedDict(foo=_y, bar=_y * 2.)
        ccddict[Q] = dict(x=copy.deepcopy(xdict), y=copy.deepcopy(ydict))

    data = dict(CCD1=copy.deepcopy(ccddict),
                CCD2=copy.deepcopy(ccddict),
                CCD3=copy.deepcopy(ccddict))

    meta = dict(suptitle='Test', doLegend=True)
    beam2dplot = BeamPlotYvX(data, meta=meta)

    beam2dplot.render(figname='')


def testBeam2ImgShow():
    """ """

    ccddict = dict()
    for iQ, Q in enumerate(['F', 'E']):
        #_x, _y = _squiggle_xy(iQ+1, iQ+1, iQ+2, iQ+2)
        #xdict = OrderedDict(foo=_x, bar=_x*2.)
        #ydict = OrderedDict(foo=_y, bar=_y*2.)
        x = np.linspace(0., 1., 10)
        y = np.linspace(0., 1., 10)
        xx, yy = np.meshgrid(x, y)
        img = np.sqrt(xx**2. + yy**2.)
        ccddict[Q] = dict(img=img)

    data = dict(CCD1=copy.deepcopy(ccddict),
                CCD2=copy.deepcopy(ccddict),
                CCD3=copy.deepcopy(ccddict))

    meta = dict(suptitle='Test')
    beamimgshow = BeamImgShow(data, meta=meta)

    beamimgshow.render(figname='')


if __name__ == '__main__':

    # testBeam2DPlot()
    testBeam2ImgShow()
