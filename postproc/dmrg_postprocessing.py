import os
import numpy as np
import json
import matplotlib.pyplot as plt
import copy
from shutil import copyfile

def LoadJSONFile(file,appendStr,funcName,count=0):
    """
    Loads data from a JSON file and corrects unfinished runs by appending a string to the file.
    """
    filemod = file[:-5]+"-mod"+".json"
    try:
        if count == 0:
            data = json.load(open(file))
        elif count == 1:
            data = json.load(open(filemod))
        else:
            raise ValueError('LoadJSONFile was not able to correct the file "{}" with an appended "{}". '
                'Check the file manually.'.format(file,appendStr))
        return data
    except json.JSONDecodeError:
        copyfile(file,filemod)
        fh = open(filemod, "a")
        fh.write(appendStr)
        fh.close()
        return LoadJSONFile(file,appendStr,funcName,count+1)

def LoadJSONDict(file):
    """
    Loads data from a JSON file with keys "headers" and "table", and corrects unfinished runs by
    appending "}".
    """
    return LoadJSONFile(file,"}","LoadJSONDict")

def LoadJSONTable(file):
    """
    Loads data from a JSON file with keys "headers" and "table", and corrects unfinished runs by
    appending "]}".
    """
    return LoadJSONFile(file,"]}","LoadJSONTable")

def LoadJSONArray(file):
    """
    Loads data from a JSON file represented as an array of dictionary entries and corrects
    unfinished runs by appending "]".
    """
    return LoadJSONFile(file,"]","LoadJSONArray")

class Data:
    """
    Post-processing of a single DMRG run
    """

    def __init__(self, base_dir, jobs_dir='', label=None, load_warmup=True, load_sweeps=True):
        """ Initializes the object.
        """
        self._base_dir = os.path.join(jobs_dir,base_dir)
        self._label = label
        self._run = None
        self._sweepIdx = None
        self._steps = None
        self._timings = None
        self._hamPrealloc = None
        self._entSpectra = None
        self._corr = None

    #
    #   Run data
    #
    def _LoadRun(self):
        if self._run is None:
                self._run = LoadJSONDict(os.path.join(self._base_dir,'DMRGRun.json'))
        return self._run

    def RunData(self):
        return self._LoadRun()

    #
    #   Steps data
    #
    def _LoadSteps(self):
        """ Loads data from DMRGSteps.json and extracts
        """
        if self._steps is None:
            steps = LoadJSONTable(os.path.join(self._base_dir,'DMRGSteps.json'))
            self._stepsHeaders = steps['headers']
            self._idxEnergy = self._stepsHeaders.index('GSEnergy')
            self._idxNSysEnl = self._stepsHeaders.index('NSites_SysEnl')
            self._idxNEnvEnl = self._stepsHeaders.index('NSites_EnvEnl')
            self._idxLoopidx = self._stepsHeaders.index('LoopIdx')
            self._idxNStatesH = self._stepsHeaders.index('NumStates_H')
            self._steps = steps['table']

    def Steps(self,header=None):
        self._LoadSteps()
        if header is None:
            return copy.deepcopy(self._steps)
        else:
            idx = self._stepsHeaders.index(header)
            return np.array([row[idx] for row in self._steps])

    def StepsHeaders(self):
        self._LoadSteps()
        return self._stepsHeaders

    def SweepIdx(self):
        if self._sweepIdx is None:
            StepIdx = self.Steps("StepIdx")
            LoopIdx = self.Steps("LoopIdx")
            self._sweepIdx = np.where(StepIdx==max(StepIdx[np.where(LoopIdx>0)]))[0]
        return self._sweepIdx

    def EnergyPerSite(self):
        self._LoadSteps()
        return np.array([row[self._idxEnergy]/(row[self._idxNSysEnl]+row[self._idxNEnvEnl]) for row in self._steps])

    def NumStatesSuperblock(self,n=None):
        self._LoadSteps()
        if n is None:
            self._NStatesH = np.array([row[self._idxNStatesH] for row in self._steps])
            return self._NStatesH
        else:
            return self._steps[n][self._idxNStatesH]

    def PlotEnergyPerSite(self,**kwargs):
        self._LoadSteps()
        energy_iter = np.array(self.EnergyPerSite())
        self._p = plt.plot(energy_iter,label=self._label,**kwargs)
        # color = self._p[-1].get_color()
        # for d in dm:
        #     plt.axvline(x=d,color=color,linewidth=1)
        self._color = self._p[-1].get_color()
        plt.xlabel('DMRG Steps')
        plt.ylabel(r'$E_0/N$')

    def PlotLoopBars(self,**kwargs):
        self._LoadSteps()
        LoopIdx = [row[self._idxLoopidx] for row in self._steps]
        dm = np.where([LoopIdx[i] - LoopIdx[i-1] for i in range(1,len(LoopIdx))])[0]
        for d in dm:
            plt.axvline(x=d,color=self._color,linewidth=1,**kwargs)

    #
    #   Timings data
    #
    def _LoadTimings(self):
        if self._timings is None:
            timings = LoadJSONTable(os.path.join(self._base_dir,'Timings.json'))
            self._timingsHeaders = timings['headers']
            self._idxTimeTot = self._timingsHeaders.index('Total')
            self._timings = timings['table']

    def Timings(self):
        self._LoadTimings()
        return copy.deepcopy(self._timings)

    def TimingsHeaders(self):
        self._LoadTimings()
        return copy.deepcopy(self._timingsHeaders)

    def TotalTime(self):
        self._LoadTimings()
        return np.array([row[self._idxTimeTot] for row in self._timings])

    def PlotTotalTime(self,**kwargs):
        totTime = self.TotalTime()
        self._p = plt.plot(totTime,label=self._label,**kwargs)
        self._color = self._p[-1].get_color()
        plt.xlabel('DMRG Steps')
        plt.ylabel('Time Elapsed per Step (s)')

    #
    #   Preallocation data
    #
    def PreallocData(self,n=None,key=None):
        if self._hamPrealloc is None:
            self._hamPrealloc = LoadJSONArray(os.path.join(self._base_dir,'HamiltonianPrealloc.json'))
        if n is None:
            return self._hamPrealloc
        else:
            if key is None:
                return self._hamPrealloc[n]
            else:
                if key=="Tnnz":
                    return np.array(self._hamPrealloc[n]["Dnnz"]) + np.array(self._hamPrealloc[n]["Onnz"])
                else:
                    return self._hamPrealloc[n][key]

    def PlotPreallocData(self,n,totals_only=True,**kwargs):
        Dnnz = np.array(self.PreallocData(n)["Dnnz"])
        Onnz = np.array(self.PreallocData(n)["Onnz"])
        self._p = plt.plot(Dnnz+Onnz,label='Tnnz: {}'.format(n),marker='o',**kwargs)
        if not totals_only:
            color = self._p[-1].get_color()
            self._p = plt.plot(Dnnz,label='Dnnz: {}'.format(n),color=color,marker='s',**kwargs)
            self._p = plt.plot(Onnz,label='Onnz: {}'.format(n),color=color,marker='^',**kwargs)

    #
    #   Entanglement Spectra
    #
    def _LoadSpectra(self):
        if self._entSpectra is None:
            spectra = LoadJSONArray(os.path.join(self._base_dir,'EntanglementSpectra.json'))
            # determine which global indices are the ends of a sweep

            self._entSpectra = spectra

    def EntanglementSpectra(self):
        ''' Loads the entanglement spectrum at the end of each sweep '''
        self._LoadSpectra()
        return [self._entSpectra[row]['Sys'] for row in self.SweepIdx()]

    def EntanglementEntropy(self):
        ''' Calculates the entanglement entropy using eigenvalues from all sectors '''
        a = self.EntanglementSpectra()
        l = [np.concatenate([a[i][j]['vals'] for j in range(len(a[i]))]) for i in range(len(a))]
        return [-np.sum( [np.log(lii)*lii  for lii in li if lii > 0] ) for li in l]

    #
    #   Correlations
    #
    def _LoadCorrelations(self):
        if self._corr is None:
            self._corr = LoadJSONTable(os.path.join(self._base_dir,'Correlations.json'))

    def Correlations(self):
        self._LoadCorrelations()
        return self._corr

class DataSeries:
    """
    Post-processing for multiple DMRG runs
    """

    def __init__(self, base_dir_list, *args, label_list=None, **kwargs):
        if label_list is None:
            self.DataList = [ Data(base_dir, *args, label=base_dir, **kwargs) for base_dir in base_dir_list ]
        else:
            self.DataList = [ Data(base_dir, *args, label=label, **kwargs) for (base_dir, label) in zip(base_dir_list,label_list) ]

    def Data(self,n):
        return self.DataList[n]

    def PlotEnergyPerSite(self,**kwargs):
        for obj in self.DataList: obj.PlotEnergyPerSite(**kwargs)

    def PlotTotalTime(self,**kwargs):
        for obj in self.DataList: obj.PlotTotalTime(**kwargs)

    def PlotLoopBars(self,n=None,**kwargs):
        if n is None:
            for obj in self.DataList: obj.PlotLoopBars(**kwargs)
        else:
            self.DataList[n].PlotLoopBars(**kwargs)
