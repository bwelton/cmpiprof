// g++ -std=c++11 ReadText.cpp -o convertbinary

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cstdint>
#include <cstring>
#include <stdio.h>
using namespace std;

class Phase{
public:
	Phase() {
		_headerSize = sizeof(uint64_t) * 7 + sizeof(double) * 5;
		_kernelSize = sizeof(double) + sizeof(uint64_t) * 4;
		_kernels.clear();
		_TotalTime = 0;
		_MallocTime = 0;
		_MallocSize = 0;
		_GPUTime = 0;
		_GPUAvg = 0;
		_MemcpyTime = 0;
		_MemcpySize = 0;
		_GTotal = 0;
		_GRead = 0;
		_GWrite = 0;
	}

	void PhaseEnd(double TotalTime, double MallocTime, uint64_t MallocSize, 
		  double GPUTime, double GPUAvg, double MemcpyTime,
		  uint64_t MemcpySize,uint64_t GTotal,uint64_t GRead, uint64_t GWrite) {
		_TotalTime = TotalTime;
		_MallocTime = MallocTime;
		_MallocSize = MallocSize;
		_GPUTime = GPUTime;
		_GPUAvg = GPUAvg;
		_MemcpyTime = MemcpyTime;
		_MemcpySize = MemcpySize;
		_GTotal = GTotal;
		_GRead = GRead;
		_GWrite = GWrite;
	}

	uint64_t HeaderSize(){
		return _headerSize;
	}

	char * ReturnHeader(uint64_t start, uint64_t end){
		char * tmp = new char[_headerSize+1];
		size_t dsize = sizeof(double);
		size_t usize = sizeof(uint64_t);
		memcpy(tmp, &_TotalTime, dsize);
		memcpy(&(tmp[dsize]), &_MallocTime, dsize);
		memcpy(&(tmp[(dsize * 2)]), &_MallocSize, usize);
		memcpy(&(tmp[(dsize * 2) + (usize * 1)]), &_GPUTime, dsize);
		memcpy(&(tmp[(dsize * 3) + (usize * 1)]), &_GPUAvg, dsize);				
		memcpy(&(tmp[(dsize * 4) + (usize * 1)]), &_MemcpyTime, dsize);
		memcpy(&(tmp[(dsize * 5) + (usize * 1)]), &_MemcpySize, usize);
		memcpy(&(tmp[(dsize * 5) + (usize * 2)]), &_GTotal, usize);
		memcpy(&(tmp[(dsize * 5) + (usize * 3)]), &_GRead, usize);
		memcpy(&(tmp[(dsize * 5) + (usize * 4)]), &_GWrite, usize);
		memcpy(&(tmp[(dsize * 5) + (usize * 5)]), &start, usize);
		memcpy(&(tmp[(dsize * 5) + (usize * 6)]), &end, usize);
		return tmp;
	}

	void addKernel(double TotalTime, uint64_t TBytesRead, uint64_t TBytesWritten, uint64_t CBytesRead, uint64_t CBytesWritten) {
		char * tmp =  new char[_kernelSize+1];
		size_t dsize = sizeof(double);
		size_t usize = sizeof(uint64_t);
		memcpy(tmp, &TotalTime, dsize);
		memcpy(&(tmp[dsize]), &TBytesRead, usize);
		memcpy(&(tmp[dsize + usize]), &TBytesWritten, usize);
		memcpy(&(tmp[dsize + (usize * 2)]), &CBytesRead, usize);
		memcpy(&(tmp[dsize + (usize * 3)]), &CBytesWritten, usize);
		_kernels.push_back(tmp);
	}

	uint64_t GetKernelLength() {
		return _kernelSize;
	}
	uint64_t GetKernelSize() {
		return _kernels.size() * _kernelSize;
	}

	char * GetNextKernel() {
		char * next = _kernels[0];
		_kernels.erase(_kernels.begin());
		return next;
	}
	uint64_t KernelsRemaining() {
		return _kernels.size();
	}

private:
	double _TotalTime;
	double _MallocTime;
	double _GPUTime;
	double _GPUAvg;
	double _MemcpyTime;
	uint64_t _MallocSize;
	uint64_t _MemcpySize;
	uint64_t _GTotal;
	uint64_t _GRead;
	uint64_t _GWrite;
	uint64_t _startPos;
	uint64_t _endPos;

	uint64_t _headerSize;
	uint64_t _kernelSize;

	std::vector<char *> _kernels;
};

//phase_end,TotalTime:0.011047,MallocTime:0.000331,MallocSize:2130790982,GPUTime:0.000000,GPUAvg:-nan,MemcpyTime:0.000000,MemcpySize:0,GTotal:0,GRead:0,GWrite:0
//kernel_exec,TotalTime:0.192032,TBytesRead:83613488,TBytesWritten:83613488,CBytesRead:28311136,CBytesWritten:28311136

int main(int argc, char *argv[])
{
	std::string infname = std::string(argv[1]);
	std::ifstream infile;
	std::vector<Phase> phases;

	string line = "";
	infile.open(infname);
	Phase CurrentPhase;
	while (!infile.eof())
	{
		std::getline(infile, line);
		if(line.compare(0,9,"phase_end") == 0){
			double TotalTime;
			double MallocTime;
			uint64_t MallocSize;

		  	double GPUTime;
		   	double GPUAvg;
		   	double MemcpyTime;

		  	uint64_t MemcpySize;
		  	uint64_t GTotal;
		  	uint64_t GRead;
		   	uint64_t GWrite;
			sscanf(line.c_str(),"phase_end,TotalTime:%lf,MallocTime:%lf,MallocSize:%llu,GPUTime:%lf,GPUAvg:%lf,MemcpyTime:%lf,MemcpySize:%llu,GTotal:%llu,GRead:%llu,GWrite:%llu",
				&TotalTime,&MallocTime,&MallocSize,&GPUTime,&GPUAvg,&MemcpyTime,
				&MemcpySize,&GTotal,&GRead,&GWrite);
			CurrentPhase.PhaseEnd(TotalTime, MallocTime, MallocSize, GPUTime, GPUAvg,
				MemcpyTime,MemcpySize,GTotal,GRead,GWrite);
			phases.push_back(CurrentPhase);
			CurrentPhase = Phase();

		} else if(line.compare(0,11,"kernel_exec") == 0) {
			double ttime;
			uint64_t tread, twrite, cread, cwrite; 			
			sscanf(line.c_str(),"kernel_exec,TotalTime:%lf,TBytesRead:%llu,TBytesWritten:%llu,CBytesRead:%llu,CBytesWritten:%llu",
				&ttime,&tread,&twrite,&cread,&cwrite);
			CurrentPhase.addKernel(ttime, tread, twrite, cread, cwrite);
			//std::cout << line << "," << ttime << "," << tread  << "," << twrite << "," << cread << "," << cwrite << std::endl;
		} else {
			std::cout << "CANT TELL: " << line << std::endl;
		}
	}
	phases.push_back(CurrentPhase);
	// Calculate offsets.


	// Write output binary file
	uint64_t count = phases.size();
	uint64_t headerSize = phases[0].HeaderSize();
	FILE *fp;
	fp=fopen(argv[2], "wb");
	fwrite(&count, sizeof(uint64_t), 1, fp);
	fwrite(&headerSize, sizeof(uint64_t), 1, fp);

	uint64_t curOffset = sizeof(uint64_t) * 2 + headerSize * count;

	for (int i = 0; i < phases.size(); i++) {
		char * tmp = phases[i].ReturnHeader(curOffset, phases[i].GetKernelSize() + curOffset);
		fwrite(tmp, sizeof(char), headerSize, fp);
		curOffset = phases[i].GetKernelSize() + curOffset;
		delete [] tmp;
	}
	for (int i = 0; i < phases.size(); i++) {
		while(phases[i].KernelsRemaining() > 0){
			char * tmp = phases[i].GetNextKernel();
			fwrite(tmp, sizeof(char), phases[i].GetKernelLength(), fp);
			delete [] tmp;
		}
	}	

	fclose(fp);
}

