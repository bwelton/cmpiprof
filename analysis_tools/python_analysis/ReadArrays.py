import struct
import sys
import os
from os import listdir
from os.path import isfile, join
import numpy as np

class ImportFile:
    def __init__(self, filename):
        self._filename = filename

    def ReadFilePart(self):
        ## Setup for reading file by phase, to (hopefully) save
        # memory. 

        ## Save a file handle somewheer
        self._file_handle = open(self._filename, "rb")
        header = self._file_handle.read(16)
        count, headerSize = struct.unpack_from('QQ',header,0)
        print "Count: " + str(count) + ",Header Size:" + str(headerSize)
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
        self._kdt = np.dtype([("TotalTime",float,1),
                     ("TBytesRead",int,1),
                     ("TBytesWritten",int,1),
                     ("CBytesRead",int,1),
                     ("CBytesWritten", int, 1)])
        self._phases = np.zeros((count), dtype=dt)
        self._kernels = []
        self._phase_offsets = []
        self._residentPhase = -1
        self.ReadHeader(count,headerSize)


    def ReadHeader(self,count, headerSize):
        # when reading file by phase, read the header into memory.
        offset = 0
        headerData = self._file_handle.read(headerSize*count)
        for i in range(0, count):
            ttime, mtime, msize, gtime, gavg, mtime, msize, gtot, gread, gwrite, start, end = struct.unpack_from("ddQdddQQQQQQ",headerData,offset)
            offset += headerSize
            self._phases[i] = (ttime, mtime, msize, gtime, gavg, mtime, msize, gtot, gread, gwrite)
            self._phase_offsets.append([start,end])

    def ReadSpecificPhase(self, phase):
        if phase >= len(self._phase_offsets):
            return np.zeros((0),dtype=self._kdt)

        ## Read the phase
        kcount = (_phase_offsets[phase][1] - _phase_offsets[phase][0]) // (8 * 5)
        self._file_handle.seek(_phase_offsets[phase][0])
        phaseData = self._file_handle.read(_phase_offsets[phase][1] - _phase_offsets[phase][0])
        tmps = np.zeros((kcount), dtype=self._kdt)
        offset = 0
        for i in range(0, kcount):
            ttime, tbytesr, tbytesw, cbytesr, cbytesw = struct.unpack_from("dQQQQ", phaseData, offset)
            offset += (8*5)
            tmps[i] = (ttime, tbytesr, tbytesw, cbytesr, cbytesw)
        return tmps

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
        tmp = np.empty_like (self._phases)
        tmp[:] = self._phases
        return tmp

    def GetKernels(self,phase):
        if len(self._kernels) < phase:
            return np.empty ((0), dtype=self._kernels[0].dtype)

        tmp = np.empty_like (self._kernels[phase])
        tmp[:] = self._kernels[phase]
        return tmp

    def GetKernelsRes(self,phase):
        if len(self._phases) <= phase:
            return np.empty ((0), dtype=self._kernels[0].dtype)
        if self._residentPhase == phase:
            return self._resident
        self._resident = self.ReadSpecificPhase(phase)
        self._residentPhase = phase
        return self._resident

    def SetKernelRest(self, phase):
        self._resident = phase


    def RemoveIOPhases(self, IOPhaseList):
        for x in IOPhaseList:
            self._phases = np.delete(self._phases, (int(x)), axis=0)            

    def FixPhase(self, expectedSize, ident=None):
        if self._phases.size != expectedSize:
            if self._phases.size > expectedSize:
                if ident != None:
                    print "- Cutting phase " + str(ident) + " from " + str(self._phases.size) + " to " + str(expectedSize)
                self._phases = np.split(self._phases,[expectedSize,self._phases.size - expectedSize])[0]
            elif self._phases.size < expectedSize:
                if ident != None:
                    print "- Expanding phase " + str(ident) + " from " + str(self._phases.size) + " to " + str(expectedSize)
                tmp = np.zeros((expectedSize-self._phases.size), dtype=self._phases.dtype)
                self._phases = np.concatenate((self._phases,tmp), axis=1)

    def ReadPhases(self, count, headerSize):
        offset = 16
        for i in range(0, count):
            ttime, mtime, msize, gtime, gavg, mtime, msize, gtot, gread, gwrite, start, end = struct.unpack_from("ddQdddQQQQQQ",self._rawData,offset)
            offset += headerSize
            self._phases[i] = (ttime, mtime, msize, gtime, gavg, mtime, msize, gtot, gread, gwrite)
            #print self._phases[i]
            self.ReadKernels(i, start, end)
        self._rawData = None


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
