import struct
import sys
import os
from os import listdir
from os.path import isfile, join
import numpy as np

class ImportFile:
    def __init__(self, filename):
        self._filename = filename

    def ReadFile(self):
        f = open(self._filename, "rb")
        self._rawData = f.read()
        f.close()
        count, headerSize = struct.unpack_from('QQ',self._rawData,0)
        #print "Count: " + str(count) + ",Header Size:" + str(headerSize)
        dt = np.dtype([("TotalTime",float,1),
                     ("MallocTime",float,1),
                     ("MallocSize",int,1),
                     ("GPUTime",float,1),
                     ("GPUAvg", float, 1),
                     ("MemcpyTime",float,1),
                     ("Memcpysize",int,1),
                     ("TotalMem",int,1),
                     ("TotalMemRead",int,1),
                     ("TotalMemWrite",int,1)])
        
        self._phases = np.zeros((count), dtype=dt)
        self._kernels = []
        self.ReadPhases(count, headerSize)

    def GetPhases(self):
        return self._phases

    def GetKernels(self):
        return self._kernels


    def ReadPhases(self, count, headerSize):
        offset = 16
        for i in range(0, count):
            ttime, mtime, msize, gtime, gavg, mtime, msize, gtot, gread, gwrite, start, end = struct.unpack_from("ddQdddQQQQQQ",self._rawData,offset)
            offset += headerSize
            self._phases[i] = (ttime, mtime, msize, gtime, gavg, mtime, msize, gtot, gread, gwrite)
            #print self._phases[i]
            self.ReadKernels(i, start, end)


    def ReadKernels(self, phase, start, end):
        kcount = (end - start ) // (8 * 5)
        #print "Phase: " + str(phase) + " has " + str(kcount) + " Kernels"
        kdt = np.dtype([("TotalTime",float,1),
                     ("TBytesRead",int,1),
                     ("TBytesWritten",int,1),
                     ("CBytesRead",int,1),
                     ("CBytesWritten", int, 1)])
        tmps = np.zeros((kcount), dtype=kdt)

        offset = start
        for i in range(0, kcount):
            ttime, tbytesr, tbytesw, cbytesr, cbytesw = struct.unpack_from("dQQQQ", self._rawData, offset)
            offset += (8*5)
            tmps[i] = (ttime, tbytesr, tbytesw, cbytesr, cbytesw)

        self._kernels.append(tmps)
        #print tmps

# trial = sys.argv[1]
# t = ImportFile(trial)
# t.ReadFile()