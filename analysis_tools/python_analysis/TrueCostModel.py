import struct
import sys
import os
from os import listdir
from os.path import isfile, join
import numpy as np
from ReadArrays import ImportFile
from GPUTransferCost import GPUTransferCost
from TransmissionDelayEstimator import TransmissionDelayEstimator

class TrueCostModel: 
    def __init__(self, importData):
        self._data = importData
        if len(importData) < 1: 
            print "ERROR NO INPUT DATA"

    def CalculateCostsPart(self,outFile):
        np.seterr(divide="warn")
        phases = self._data[0].GetPhases()
        expectedSize = phases.size
        MaxCPUTimes = np.zeros((expectedSize), dtype=float)
        AvgCPUTime = np.zeros((expectedSize), dtype=float)
        CorrectedCPUTimes = np.zeros((expectedSize), dtype=float)
        ## Local copy of phases
        phases = []
        for x in self._data:
            phases.append(x.GetPhases())

        ## Sum CPU times to find average for all phases
        for x in phases:
            AvgCPUTime = AvgCPUTime + x["TotalTime"]

        AvgCPUTime =  AvgCPUTime / len(self._data)
        GPUCostEstimate = GPUTransferCost()
        transmissionCost = TransmissionDelayEstimator()
        TotalTime = 0.0
        FinalTime = 0.0
        outStr = "BeforeLB,GPUBefore,GPUAfter,AfterLB,NumberOfMigrations\n"
        ##
        # No queue depth, on demand. 
        for x in range(0, expectedSize):
            print "Processing Phase " + str(x)
            phaseList = np.zeros((len(phases)), dtype=float)
            beforeGPUList = np.zeros((len(phases)), dtype=float)
            for y in range(0,len(phases)):
                phaseList[y] = phases[y]["TotalTime"][x]
                beforeGPUList[y] = phases[y]["GPUTime"][x]

            print "-- Before Max Time: " + str(np.amax(phaseList))
            beforeMax = np.amax(phaseList)
            outStr += str(beforeMax) + "," + str(np.amax(beforeGPUList)) + ","

            TotalTime = TotalTime + beforeMax
            done = False
            migratedList = {}
            migrationCount = 0
            while done == False:
                ## Sort the Array, last element is largest
                sortArray = np.argsort(phaseList)

                myPos = sortArray[-1]
                if myPos not in migratedList:
                    migratedList[myPos] = {}


                ## Grab the GPU calls of the slowest node
                try:
                    GPUKernels = self._data[sortArray[-1]].GetKernelsRes(x)
                except:
                    print sortArray
                    print len(self._data)
                    break
                ## Determine Migration costs for all kernels, 
                bytesRead = GPUCostEstimate.EstimateCost(GPUKernels["TBytesRead"])
                bytesWritten = GPUCostEstimate.EstimateCost(GPUKernels["TBytesWritten"]) 
                transferCost =  bytesWritten["ToGPU"] + bytesWritten["FromGPU"] + bytesRead["ToGPU"] + bytesWritten["FromGPU"]

                ## Add NetworkDelay, Titan nearest used right now.
                transferCost = transferCost + transmissionCost.EstimateDelay(GPUKernels["TBytesRead"])["Titan_Nearest"] + transmissionCost.EstimateDelay(GPUKernels["TBytesWritten"])["Titan_Nearest"]

                ## Add GPU Computation time to mix
                transferCost = GPUKernels["TotalTime"]

                gpuSorted = np.argsort(transferCost)
                for y in range(0,len(gpuSorted)):
                    ## If the new runtime is less than us with this transfer, perform it.
                    if transferCost[gpuSorted[y]] + phaseList[sortArray[0]] + GPUKernels["TotalTime"][y] < phaseList[myPos] and gpuSorted[y] not in migratedList[myPos]:
                        #print "Transfering : " + str(myPos) + ":" + str(sortArray[0]) 
                        phaseList[myPos] = phaseList[myPos] - GPUKernels["TotalTime"][y]
                        beforeGPUList[myPos] = beforeGPUList[myPos] - GPUKernels["TotalTime"][y]
                        phaseList[sortArray[0]] = phaseList[sortArray[0]] + transferCost[gpuSorted[y]] + GPUKernels["TotalTime"][y]
                        beforeGPUList[sortArray[0]] = beforeGPUList[sortArray[0]] + GPUKernels["TotalTime"][y] + transferCost[gpuSorted[y]] 
                        sortArray = np.argsort(phaseList)
                        migratedList[myPos][gpuSorted[y]] = 1
                        migrationCount += 1                        
                    else: 
                        break
                sortArray = np.argsort(phaseList)
                ## If we could not improve, we are done.
                if sortArray[-1] == myPos:
                    done = True
                    break
            afterMax = np.amax(phaseList)
            FinalTime = afterMax + FinalTime
            print "-- New Maximum Time: " + str(afterMax)
            outStr += str(np.amax(beforeGPUList)) + "," + str(afterMax) + "," + str(migrationCount) + "\n"
            print "-- Time Saved: " + str(beforeMax - afterMax)
        print "Total Runtime: " + str(TotalTime)
        print "Final Runtime: " + str(FinalTime)
        f = open(outFile,"wb")
        f.write(outStr)
        f.close()

    def CalculateCosts(self,outFile):
        np.seterr(divide="warn")
        phases = self._data[0].GetPhases()
        expectedSize = phases.size
        MaxCPUTimes = np.zeros((expectedSize), dtype=float)
        AvgCPUTime = np.zeros((expectedSize), dtype=float)
        CorrectedCPUTimes = np.zeros((expectedSize), dtype=float)

        ## Local copy of phases
        phases = []
        for x in self._data:
            phases.append(x.GetPhases())

        ## Sum CPU times to find average for all phases
        for x in phases:
            AvgCPUTime = AvgCPUTime + x["TotalTime"]

        AvgCPUTime =  AvgCPUTime / len(self._data)
        GPUCostEstimate = GPUTransferCost()
        transmissionCost = TransmissionDelayEstimator()
        TotalTime = 0.0
        FinalTime = 0.0
        outStr = "BeforeLB,GPUBefore,GPUAfter,AfterLB,NumberOfMigrations\n"
        ##
        # No queue depth, on demand. 
        for x in range(0, expectedSize):
            print "Processing Phase " + str(x)
            phaseList = np.zeros((len(phases)), dtype=float)
            beforeGPUList = np.zeros((len(phases)), dtype=float)
            for y in range(0,len(phases)):
                phaseList[y] = phases[y]["TotalTime"][x]
                beforeGPUList[y] = phases[y]["GPUTime"][x]

            print "-- Before Max Time: " + str(np.amax(phaseList))
            beforeMax = np.amax(phaseList)
            outStr += str(beforeMax) + "," + str(np.amax(beforeGPUList)) + ","

            TotalTime = TotalTime + beforeMax
            done = False
            migratedList = {}
            migrationCount = 0
            while done == False:
                ## Sort the Array, last element is largest
                sortArray = np.argsort(phaseList)

                myPos = sortArray[-1]
                if myPos not in migratedList:
                    migratedList[myPos] = {}


                ## Grab the GPU calls of the slowest node
                try:
                    GPUKernels = self._data[sortArray[-1]].GetKernels(x)
                except:
                    print sortArray
                    print len(self._data)
                    break
                ## Determine Migration costs for all kernels, 
                bytesRead = GPUCostEstimate.EstimateCost(GPUKernels["TBytesRead"])
                bytesWritten = GPUCostEstimate.EstimateCost(GPUKernels["TBytesWritten"]) 
                transferCost =  bytesWritten["ToGPU"] + bytesWritten["FromGPU"] + bytesRead["ToGPU"] + bytesWritten["FromGPU"]

                ## Add NetworkDelay, Titan nearest used right now.
                transferCost = transferCost + transmissionCost.EstimateDelay(GPUKernels["TBytesRead"])["Titan_Nearest"] + transmissionCost.EstimateDelay(GPUKernels["TBytesWritten"])["Titan_Nearest"]

                ## Add GPU Computation time to mix
                transferCost = GPUKernels["TotalTime"]

                gpuSorted = np.argsort(transferCost)
                for y in range(0,len(gpuSorted)):
                    ## If the new runtime is less than us with this transfer, perform it.
                    if transferCost[gpuSorted[y]] + phaseList[sortArray[0]] + GPUKernels["TotalTime"][y] < phaseList[myPos] and gpuSorted[y] not in migratedList[myPos]:
                        #print "Transfering : " + str(myPos) + ":" + str(sortArray[0]) 
                        phaseList[myPos] = phaseList[myPos] - GPUKernels["TotalTime"][y]
                        beforeGPUList[myPos] = beforeGPUList[myPos] - GPUKernels["TotalTime"][y]
                        phaseList[sortArray[0]] = phaseList[sortArray[0]] + transferCost[gpuSorted[y]] + GPUKernels["TotalTime"][y]
                        beforeGPUList[sortArray[0]] = beforeGPUList[sortArray[0]] + GPUKernels["TotalTime"][y] + transferCost[gpuSorted[y]] 
                        sortArray = np.argsort(phaseList)
                        migratedList[myPos][gpuSorted[y]] = 1
                        migrationCount += 1
                    else: 
                        break
                sortArray = np.argsort(phaseList)
                ## If we could not improve, we are done.
                if sortArray[-1] == myPos:
                    done = True
                    break
            afterMax = np.amax(phaseList)
            FinalTime = afterMax + FinalTime
            print "-- New Maximum Time: " + str(afterMax)
            outStr += str(np.amax(beforeGPUList)) + "," + str(afterMax) + "," + str(migrationCount) + "\n"
            print "-- Time Saved: " + str(beforeMax - afterMax)
        print "Total Runtime: " + str(TotalTime)
        print "Final Runtime: " + str(FinalTime)
        f = open(outFile,"wb")
        f.write(outStr)
        f.close()
        
