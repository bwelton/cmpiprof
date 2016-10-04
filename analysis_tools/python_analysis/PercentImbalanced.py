import struct
import sys
import os
from os import listdir
from os.path import isfile, join
import numpy as np
from ReadArrays import ImportFile
##
# Calculate the percent imbalanced between nodes
class PercentImbalance:
    def __init__(self, importData):
        self._data = importData
        if len(importData) < 1: 
            print "ERROR NO INPUT DATA"

    def Calculate(self, outFile):
        phases = self._data[0].GetPhases()
        print len(phases)
        expectedSize = phases.size
        total = np.zeros((expectedSize), dtype=float)
        maxValues = np.zeros((expectedSize), dtype=float)      
        for x in self._data:
            phase = x.GetPhases()
            if phase.size != expectedSize:
                print "A phase is not of expected size"
                if phase.size > expectedSize:
                    print "Cutting array down"
                    phase = np.split(phase,[expectedSize,phase.size - expectedSize])[0]
                elif phase.size < expectedSize:
                    print "Expanding phase size"
                    tmp = np.zeros((expectedSize-phase.size), dtype=phases.dtype)
                    phase = np.concatenate((phase,tmp), axis=1)
                    print phase
            x.FixPhase(expectedSize)
            ## Update totals
            print len(total)
            print len(phase["GPUTime"])
            total = total + phase["GPUTime"]
            maxValues = np.maximum(phase["GPUTime"], maxValues)

        print total
        print maxValues
        maxValues = maxValues * len(self._data)
        np.seterr(divide="warn")
        finalCost = total / maxValues
        f = open(outFile,"wb")
        f.write("Phase,Percent Imbalanced\n")
        count = 1
        for x in finalCost:
            f.write(str(count) + "," + str(x) +"\n")
            count += 1

        f.close()











