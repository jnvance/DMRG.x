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

    def __init__(self, base_dir, load_warmup=True, load_sweeps=True):

        self.base_dir = base_dir

        Steps = LoadJSONTable(os.path.join(base_dir,'DMRGSteps.json'))
        self.StepsHeaders = Steps['headers']
        self.idxEnergy = self.StepsHeaders.index('GSEnergy')
        self.idxNSysEnl = self.StepsHeaders.index('NSites_SysEnl')
        self.idxNEnvEnl = self.StepsHeaders.index('NSites_EnvEnl')
        self.idxLoopidx = self.StepsHeaders.index('LoopIdx')
        self.Steps = Steps['table']

        Timings = LoadJSONTable(os.path.join(base_dir,'Timings.json'))
        self.TimingsHeaders = Timings['headers']
        self.idxTimeTot = self.TimingsHeaders.index('Total')
        self.Timings = Timings['table']

    def EnergyPerSite(self):
        return [row[self.idxEnergy]/(row[self.idxNSysEnl]+row[self.idxNEnvEnl]) for row in self.Steps]


    def TotalTime(self):
        return [row[self.idxTimeTot] for row in self.Timings]


    def PlotEnergyPerSite(self,**kwargs):
        energy_iter = np.array(self.EnergyPerSite())
        self.p = plt.plot(energy_iter,**kwargs)
        # color = self.p[-1].get_color()
        # for d in dm:
        #     plt.axvline(x=d,color=color,linewidth=1)
        plt.xlabel('DMRG Steps')
        plt.ylabel(r'$E_0/N$')


    def PlotTotalTime(self,**kwargs):
        totTime = self.TotalTime()
        self.p = plt.plot(totTime,**kwargs)
        plt.xlabel('DMRG Steps')
        plt.ylabel('Time Elapsed per Step (s)')


    def LoopBars(self,**kwargs):
        LoopIdx = [row[self.idxLoopidx] for row in self.Steps]
        dm = np.where([LoopIdx[i] - LoopIdx[i-1] for i in range(1,len(LoopIdx))])[0]
        color = self.p[-1].get_color()
        for d in dm:
            plt.axvline(x=d,color=color,linewidth=1,**kwargs)

