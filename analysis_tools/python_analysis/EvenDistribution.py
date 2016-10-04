import struct
import sys
import os
from os import listdir
from os.path import isfile, join
import numpy as np
from ReadArrays import ImportFile
from GPUTransferCost import GPUTransferCost
from TransmissionDelayEstimator import TransmissionDelayEstimator

class EvenDistribution: 
    def __init__(self, importData):
        self._data = importData
        if len(importData) < 1: 
            print "ERROR NO INPUT DATA"


    def RunEvenDistribution(self, outFile):
        np.seterr(divide="warn")
        phases = self._data[0].GetPhases()
        expectedSize = phases.size

        ## Make a copy of all phases
        phases = []
        for x in self._data:
            phases.append(x.GetPhases())

        ## Compute Average Total Time:
        AvgCPUTime = np.zeros((expectedSize), dtype=float)
        for x in phases:
            AvgCPUTime = AvgCPUTime + x["TotalTime"]

        AvgCPUTime =  AvgCPUTime / len(self._data)
        
        TotalMax = 0.0
        TotalFinal = 0.0
        GPUCostEstimate = GPUTransferCost()
        transmissionCost = TransmissionDelayEstimator()

        outStr = "BeforeLB,Target,AfterLB,NumOfMigrations\n"
        for x in range(0, expectedSize):
            phaseList = np.zeros((len(phases)), dtype=float)
            print "Processing Phase " + str(x)
            for y in range(0,len(phases)):
                phaseList[y] = phases[y]["TotalTime"][x]
            beforeMax = np.amax(phaseList)
            TotalMax = TotalMax + beforeMax
            target = AvgCPUTime[x]
            print "-- Before Max Time: " + str(np.amax(phaseList))
            print "-- Target Time Per Phase: " + str(target)
            outStr += str(beforeMax) + "," + str(target) + ","
            ## Find excess work from all nodes
            ExcessWork = np.empty((0), dtype=self._data[0].GetKernels(0).dtype)
            OriginalNode = np.empty((0), dtype=int)
            for y in range(0,len(phases)):
                if phases[y]["TotalTime"][x] > target:
                    kernels = self._data[y].GetKernels(x)
                    ## Remove the excess work
                    phases[y]["TotalTime"][x] = phases[y]["TotalTime"][x] - np.sum(kernels["TotalTime"])
                    for z in kernels:
                        if phases[y]["TotalTime"][x] + z["TotalTime"] > target:
                            ExcessWork = np.append(ExcessWork, z)
                            OriginalNode = np.append(OriginalNode,y)
            bytesRead = GPUCostEstimate.EstimateCost(ExcessWork["TBytesRead"])
            bytesWritten = GPUCostEstimate.EstimateCost(ExcessWork["TBytesWritten"]) 
            transferCost =  bytesWritten["ToGPU"] + bytesWritten["FromGPU"] + bytesRead["ToGPU"] + bytesWritten["FromGPU"]
            transferCost = transferCost + transmissionCost.EstimateDelay(ExcessWork["TBytesRead"])["Titan_Nearest"] + transmissionCost.EstimateDelay(ExcessWork["TBytesWritten"])["Titan_Nearest"]
            
            migrationCount = 0
            ## Migrate excess work:
            for y in range(0,len(phases)):
                if phases[y]["TotalTime"][x] < target:
                    deletevals = []
                    for z in range(0, len(ExcessWork)):
                        if ExcessWork["TotalTime"][z] + transferCost[z] + phases[y]["TotalTime"][x] < target:
                            phases[y]["TotalTime"][x] = ExcessWork["TotalTime"][z] + transferCost[z] + phases[y]["TotalTime"][x] 
                            deletevals.append(z)
                    for z in deletevals:
                        migrationCount += 1
                        ExcessWork = np.delete(ExcessWork, z)
                        transferCost = np.delete(transferCost,z)
                        OriginalNode = np.delete(OriginalNode,z)
            afterMax = 0.0
            if len(ExcessWork) == 0:
                phaseList = np.zeros((len(phases)), dtype=float)
                for y in range(0,len(phases)):
                    phaseList[y] = phases[y]["TotalTime"][x]
                afterMax = np.amax(phaseList)
                print "- Phase Balanced, New Phase time: " +  str(afterMax)
                outStr += str(afterMax) + "," + str(migrationCount) + "\n"
            else: 
                ## We need a second pass to try and place work as optimally as possible:
                phaseList = np.zeros((len(phases)), dtype=float)
                for y in range(0,len(phases)):
                    phaseList[y] = phases[y]["TotalTime"][x]
                sortArray = np.argsort(phaseList)                
                for z in range(0, len(ExcessWork)):
                    cheapestTransfer = ExcessWork["TotalTime"][z] + transferCost[z] + phases[sortArray[-1]]["TotalTime"][x] 
                    if cheapestTransfer < phases[OriginalNode[z]]["TotalTime"][x] + ExcessWork["TotalTime"][z]:
                        phases[sortArray[-1]]["TotalTime"][x] = ExcessWork["TotalTime"][z] + transferCost[z] + phases[sortArray[-1]]["TotalTime"][x] 
                        phaseList[sortArray[-1]] =  phases[sortArray[-1]]["TotalTime"][x] 
                        migrationCount += 1 
                    else:
                        phases[OriginalNode[z]]["TotalTime"][x] += ExcessWork["TotalTime"][z]
                        phaseList[OriginalNode[z]] =  phases[OriginalNode[z]]["TotalTime"][x] 
                    sortArray = np.argsort(phaseList)
                afterMax = np.amax(phaseList)
                print "- Could Not Meet Target, New Phase time: " +  str(afterMax)  
                outStr += str(afterMax) + "," + str(migrationCount) + "\n"
            print "- Time Saved: " + str(beforeMax - afterMax)
            TotalFinal += afterMax
        print "Before Total Time: " + str(TotalMax) + " After Final Time: " + str(TotalFinal)
        f = open(outFile,"wb")
        f.write(outStr)
        f.close()
                ## Add NetworkDelay, Titan nearest used right now.
                




#     ## First Stripe, Find Excess Work.
#     ExcessWork = []
#     for y in range(0,len(phases)):
#         ## If the work that was originall
#         if phases[y]["TotalTime"][x] + np.sum(tmpGPUs[y]["TotalTime"]) < target:
#             phases[y]["TotalTime"][x] = phases[y]["TotalTime"][x] + np.sum(tmpGPUs[y]["TotalTime"]) 
#         else:

#         while phases[y]["TotalTime"][x] < target:







# ## Algorithm
# # Remove all GPU kernel times from all nodes. 

# MaxCPUTimes = np.zeros((expectedSize), dtype=float)
# AvgCPUTime = np.zeros((expectedSize), dtype=float)
# CorrectedCPUTimes = np.zeros((expectedSize), dtype=float)        
