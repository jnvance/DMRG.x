import os
import numpy as np
import json
import matplotlib.pyplot as plt
from shutil import copyfile

warnings = True

def LoadJSONTable(file):
    try:
        data = json.load(open(file))
        return data
    except json.JSONDecodeError:
        # most likely the run did not complete
        # and the file needs to be closed by "]}"
        origcopy = file+".orig"
        copyfile(file,origcopy)
        if warnings:
            print("LoadJSONTable: File \"{}\" copied to \"{}\" and overwritten.".format(file, origcopy))
        fh = open(file, "a")
        fh.write("]}")
        fh.close()
        return LoadJSONTable(file)


class Data:
    'Post-processing a single DMRG run'

    def __init__(self, base_dir, jobs_dir='', label=None, load_warmup=True, load_sweeps=True, ):

        self._base_dir = os.path.join(jobs_dir,base_dir)

        steps = LoadJSONTable(os.path.join(self._base_dir,'DMRGSteps.json'))
        self._stepsHeaders = steps['headers']
        self._idxEnergy = self._stepsHeaders.index('GSEnergy')
        self._idxNSysEnl = self._stepsHeaders.index('NSites_SysEnl')
        self._idxNEnvEnl = self._stepsHeaders.index('NSites_EnvEnl')
        self._idxLoopidx = self._stepsHeaders.index('LoopIdx')
        self._steps = steps['table']
        self._label = label

        Timings = LoadJSONTable(os.path.join(self._base_dir,'Timings.json'))
        self.TimingsHeaders = Timings['headers']
        self._idxTimeTot = self.TimingsHeaders.index('Total')
        self.Timings = Timings['table']

    def EnergyPerSite(self):
        return np.array([row[self._idxEnergy]/(row[self._idxNSysEnl]+row[self._idxNEnvEnl]) for row in self._steps])

    def TotalTime(self):
        return np.array([row[self._idxTimeTot] for row in self.Timings])

    def PlotEnergyPerSite(self,**kwargs):
        energy_iter = np.array(self.EnergyPerSite())
        self._p = plt.plot(energy_iter,label=self._label,**kwargs)
        # color = self._p[-1].get_color()
        # for d in dm:
        #     plt.axvline(x=d,color=color,linewidth=1)
        self._color = self._p[-1].get_color()
        plt.xlabel('DMRG Steps')
        plt.ylabel(r'$E_0/N$')

    def PlotTotalTime(self,**kwargs):
        totTime = self.TotalTime()
        self._p = plt.plot(totTime,label=self._label,**kwargs)
        self._color = self._p[-1].get_color()
        plt.xlabel('DMRG Steps')
        plt.ylabel('Time Elapsed per Step (s)')

    def PlotLoopBars(self,**kwargs):
        LoopIdx = [row[self._idxLoopidx] for row in self._steps]
        dm = np.where([LoopIdx[i] - LoopIdx[i-1] for i in range(1,len(LoopIdx))])[0]
        for d in dm:
            plt.axvline(x=d,color=self._color,linewidth=1,**kwargs)

class DataSeries:
    'Post-processing for multiple DMRG runs'

    def __init__(self, base_dir_list, *args, label_list=None, **kwargs):
        if label_list==None:
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
