import struct
import sys
import os
from os import listdir
from os.path import isfile, join
import numpy as np
from ReadArrays import ImportFile

##
# Calculate the percent imbalanced between nodes
class BestCaseRebalancing:
    def __init__(self, importData):
        self._data = importData
        if len(importData) < 1: 
            print "ERROR NO INPUT DATA"

    ## Simple averaging of all nodes, absolute best case rebalancing.
    # No transfer cost associated with this. 
    # This calculation is essentially CPUTime - GPUTime + AvgGPUTime for every
    # phase
    def SimpleAverage(self):
    	np.seterr(divide="warn")
    	phases = self._data[0].GetPhases()
        expectedSize = phases.size
        GPUTotal = np.zeros((expectedSize), dtype=float)
        MaxCPUTimes = np.zeros((expectedSize), dtype=float)
        MaxGPUTimes = np.zeros((expectedSize), dtype=float)
        for x in self._data:
        	GPUTotal = GPUTotal + x.GetPhases()["GPUTime"]
        	MaxCPUTimes = np.maximum(x.GetPhases()["TotalTime"], MaxCPUTimes)
        	MaxGPUTimes = np.maximum(x.GetPhases()["GPUTime"], MaxGPUTimes)

        GPUTotal = GPUTotal / len(self._data)
        final = MaxCPUTimes - MaxGPUTimes + GPUTotal
        timeSaved = MaxCPUTimes - final

        print "Maximum benefit per phase:"
        for x in timeSaved:
        	print "Phase Time Saved: " + str(x)

        print "Total time savings: " + str(np.sum(timeSaved))
        print "Percent of time saved: " + str((np.sum(timeSaved)/np.sum(MaxCPUTimes)) * 100)
