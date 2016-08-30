/* Copyright Benjamin Welton 2015 */

#ifndef __CUDA_INTERCEPT
#define __CUDA_INTERCEPT 1

#define __dv(v) = v

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <dlfcn.h>
#include <vector>
#include "cuda_runtime_api.h"
#include "crt/host_runtime.h"
#include <sys/time.h>
#include <set>
#include <memory>
#include <map>
#include <iostream>
#include <thread>
#include <chrono>
#include <mutex>
#include <sstream>
#include <fstream>
#include <unistd.h>

double diffTime(timeval start, timeval end);

enum AsyncType { KERNEL, MEMCPY };
struct ActiveTimer;

struct KernelMemCounter {
	uint64_t total_read;
	uint64_t total_write;
	uint64_t changed_read;
	uint64_t changed_write;
};

struct PhaseTimers {
	std::vector<KernelMemCounter> kernel_mem_info;
	std::vector<std::pair<cudaEvent_t,cudaEvent_t> > gpu_timers;
	std::vector<std::pair<cudaEvent_t,cudaEvent_t> > memory_timers;
	size_t mem_transfer_size;
	size_t mem_alloc_size;
	double mem_alloc_time;
	double mem_transfer_time;
	double gpu_exe_count;
	double gpu_exe_avg;
	double gpu_exe_time;
	double phase_time;
	int timer_count;
	bool phase_started;
	uint64_t readb;
	uint64_t writeb;
	uint64_t totalb;
};

struct KernelExecution {
	dim3 gridDim;
	dim3 blockDim;
	size_t sharedMem;
	cudaStream_t stream;
	std::set<const void *> StoreArgPush;
	float execution_ms;
	std::string func_name;
	ActiveTimer * timer;
};

struct AsyncMemcpy {
	void *dst;
	const void *src;
	size_t count;
	enum cudaMemcpyKind direction;
	ActiveTimer * timer;
};

struct ActiveTimer {
	enum AsyncType type;
	std::pair<cudaEvent_t,cudaEvent_t> p;
	KernelExecution * kernel;
	AsyncMemcpy * aMemcpy;
};

class PerfStorage {
public:
	PerfStorage() {
		char hostname2[1024];
		hostname2[1023] = '\0';
		gethostname(hostname2, 1023);
		std::thread::id this_id = std::this_thread::get_id();
		std::cout << "Creating perf storage for thread " << this_id << "\n";
		std::cout << "My hostname: " << hostname2 << "\n";   
		_currentExecution = (KernelExecution *) NULL;
		_currentPhase = new PhaseTimers[1];
		ZeroPhase(_currentPhase);
		_thread_id = this_id;
		_timesCalled = 0;
		_cur_stream = 0;
		char hostname[128];
		hostname[127] = '\000';
		gethostname(hostname, 128);
		std::stringstream getFilename;
		getFilename << hostname << "." << _thread_id << ".out";
		outFile.open(getFilename.str());
		gettimeofday(&last_write, NULL);	
		gettimeofday(&begin_phase, NULL);
		uni_counter = 0;
		last_kernelexec = 0;
		arg_stack.clear();
		_currentPhase->kernel_mem_info.clear();
	}
	~PerfStorage();

	void LogEntry(char * entry);

	double TimerTotal(std::vector<std::pair<cudaEvent_t,cudaEvent_t> > & t, bool kernelT);
	void ZeroPhase(PhaseTimers * phase);
	void DeleteCudaTimers(std::vector<std::pair<cudaEvent_t,cudaEvent_t> > & t);
	void CheckTimers(bool phase_end);
	void AddMemoryTimer(cudaEvent_t start, cudaEvent_t end, size_t size);
	void AddGPUTimer(cudaEvent_t start, cudaEvent_t end);
	void MallocTime(cudaEvent_t start, cudaEvent_t end, size_t size, void * ptr);
	void BeginPhase();
	void EndPhase();
	void WritePhasePart();
	cudaStream_t GetStream();
	bool CheckThread();
	void SetStream(cudaStream_t st);
	struct timeval begin_phase, end_phase, last_write;
	void emergancyShutdown();
	void AddMemRead(void * mem_loc);
	void AddMemWrite(void * mem_loc);
	void DeleteMem(void * mem_loc);
	void PushArgument(void * mem_loc);
	void LaunchedKernel();
	void * FindPtr(void * mem_loc);
	void LogKernelInfo(float elapsed_time);
private:
	std::ofstream outFile;
	PhaseTimers * _currentPhase;
	std::vector<KernelExecution *> _streamKernels;
	std::vector<AsyncMemcpy *> _asyncMemcpys;
	std::set<ActiveTimer *> _timers;
	size_t _timesCalled;
	KernelExecution * _currentExecution;
	std::map<void *, size_t> _deviceMemory;
	std::map<cudaStream_t, std::pair<cudaEvent_t,cudaEvent_t> > _streamTimers;
	std::map<void *, std::string> _cubinMap; 
	std::thread::id _thread_id;
	std::stringstream  _log;
	std::string _cur_cudaFunc;
	cudaStream_t _cur_stream;

	uint64_t uni_counter;
	uint64_t last_kernelexec;
	std::map<void *, std::pair<uint64_t,uint64_t> > _mem_addrs;
	std::map<void *, size_t > _mem_size;
	std::vector<void *> arg_stack;


};

thread_local std::shared_ptr<PerfStorage> PerfStorageDataClass;
void PerfStorage::SetStream(cudaStream_t st){
	_cur_stream = st;
}
cudaStream_t PerfStorage::GetStream(){
	cudaStream_t tmp = _cur_stream;
	// _cur_stream = 0;
	return tmp;
}

void PerfStorage::emergancyShutdown(){
	_mem_addrs.clear();
	_mem_size.clear();
	arg_stack.clear();
	_currentPhase->gpu_timers.clear();
	_currentPhase->memory_timers.clear();
}


bool PerfStorage::CheckThread(){
	if (_thread_id == std::this_thread::get_id())
		return true;
	return false;
}

void PerfStorage::DeleteCudaTimers(std::vector<std::pair<cudaEvent_t,cudaEvent_t> > & t) {
	for (int x = 0; x < t.size(); x++) {
		cudaEventDestroy(t[x].first);
		cudaEventDestroy(t[x].second);
	}
}

void PerfStorage::ZeroPhase(PhaseTimers * phase){
	DeleteCudaTimers(phase->gpu_timers);
	DeleteCudaTimers(phase->memory_timers);

	phase->gpu_timers.clear();
	phase->memory_timers.clear();
	phase->kernel_mem_info.clear();
	phase->mem_transfer_size = 0;
	phase->mem_alloc_size = 0;
	phase->mem_alloc_time = 0.0;
	phase->mem_transfer_time = 0.0;
	phase->gpu_exe_count = 0.0;
	phase->gpu_exe_avg = 0.0;
	phase->phase_time = 0.0;
	phase->timer_count = 0;
	phase->gpu_exe_time = 0.0;
	phase->phase_started = false;
	phase->totalb = 0;
	phase->readb = 0;
	phase->writeb = 0;

}

double PerfStorage::TimerTotal(std::vector<std::pair<cudaEvent_t,cudaEvent_t> > & t, bool kernelT){
	// Assumes synchronized device
	if (CheckThread()==false){
		fprintf(stderr, "%s\n", "We Have failed the thread check");
	}
	float exe_time = 0.0;
	for (int x = 0; x < t.size(); x++) {
		float tmp = 0.0;
		cudaEventElapsedTime(&tmp, t[x].first, t[x].second);
		exe_time = exe_time + tmp;
		if (kernelT == true){
			LogKernelInfo(tmp / 1000);
		}
	}
	return double(exe_time / 1000);	
}


void PerfStorage::LogKernelInfo(float elapsed_time) {
	// Log this kernel entry into the database	
	if (_currentPhase->kernel_mem_info.size() == 0){
		fprintf(stderr, "%s\n", "We don't have a kernel exectuion");
		return;
	}
	char tmp[1500];
	snprintf(tmp, 1500, "kernel_exec,TotalTime:%f,TBytesRead:%llu,TBytesWritten:%llu,CBytesRead:%llu,CBytesWritten:%llu",
			 elapsed_time,_currentPhase->kernel_mem_info[0].total_read,_currentPhase->kernel_mem_info[0].total_write,_currentPhase->kernel_mem_info[0].changed_read, 
			 _currentPhase->kernel_mem_info[0].changed_write );
	LogEntry(tmp);
	_currentPhase->kernel_mem_info.erase(_currentPhase->kernel_mem_info.begin());
}

void PerfStorage::CheckTimers(bool phase_end){
	if (phase_end == false && _currentPhase->timer_count < 50)
		return;
	cudaDeviceSynchronize();

	double gpu_timer = TimerTotal(_currentPhase->gpu_timers, true);
	double memory_timer = TimerTotal(_currentPhase->memory_timers, false);

	_currentPhase->mem_transfer_time = _currentPhase->mem_transfer_time + memory_timer;
	_currentPhase->gpu_exe_time = _currentPhase->gpu_exe_time + gpu_timer;
	_currentPhase->gpu_exe_count = _currentPhase->gpu_exe_count + _currentPhase->gpu_timers.size();

	DeleteCudaTimers(_currentPhase->gpu_timers);
	DeleteCudaTimers(_currentPhase->memory_timers);
	_currentPhase->kernel_mem_info.clear();
	_currentPhase->gpu_timers.clear();
	_currentPhase->memory_timers.clear();
	_currentPhase->timer_count = 0;

	if (phase_end == false) {
		WritePhasePart();
	}
}



void PerfStorage::AddMemoryTimer(cudaEvent_t start, cudaEvent_t end, size_t size) {
	_currentPhase->phase_started = true;

	std::pair<cudaEvent_t,cudaEvent_t> p;
	p.first = start;
	p.second = end;
	_currentPhase->memory_timers.push_back(p);
	_currentPhase->timer_count++;
	_currentPhase->mem_transfer_size = _currentPhase->mem_transfer_size + size;
	CheckTimers(true);
}

void PerfStorage::AddGPUTimer(cudaEvent_t start, cudaEvent_t end) {
	_currentPhase->phase_started = true;

	std::pair<cudaEvent_t,cudaEvent_t> p;
	p.first = start;
	p.second = end;
	_currentPhase->gpu_timers.push_back(p);
	_currentPhase->timer_count++;
	CheckTimers(false);	
}

void PerfStorage::MallocTime(cudaEvent_t start, cudaEvent_t end, size_t size, void * ptr) {
	_currentPhase->phase_started = true;

	float tmp = 0.0;
	cudaEventElapsedTime(&tmp, start, end);	
	_currentPhase->mem_alloc_time = _currentPhase->mem_alloc_time + double(tmp / 1000);
	_currentPhase->mem_alloc_size = _currentPhase->mem_alloc_size + size;
	cudaEventDestroy(start);
	cudaEventDestroy(end);

	uni_counter++;
	_mem_addrs[ptr] = std::make_pair(uni_counter, uni_counter);
	_mem_size[ptr] = size;
}


void PerfStorage::AddMemRead(void * mem_loc){
	uni_counter++;

	if (_mem_size.find(mem_loc) == _mem_size.end()){
		void * trueMem = FindPtr(mem_loc);
		if (trueMem == NULL){
			fprintf(stderr, "%s: %p\n", "error finding WRITE mem location for", mem_loc);
			_mem_size[mem_loc] = 0; 
			_mem_addrs[mem_loc] = std::make_pair(uni_counter,uni_counter);
		} else {
			_mem_addrs[trueMem].first = uni_counter;	
		}
	} else {
		_mem_addrs[mem_loc].first = uni_counter;
	}
}

void PerfStorage::DeleteMem(void * mem_loc){
	if (_mem_size.find(mem_loc) != _mem_size.end()) {
		_mem_size.erase(mem_loc);
		_mem_addrs.erase(mem_loc);
	}
}

void PerfStorage::LaunchedKernel() {
	size_t totalBytes = 0;
	size_t readBytes = 0;
	size_t writeBytes = 0;

	KernelMemCounter loc;
	uint64_t read_unchanged = 0;
	uint64_t write_unchanged = 0;
	loc.total_read = 0;
	loc.total_write = 0;
	loc.changed_read = 0;
	loc.changed_write = 0;

	uni_counter++;
	for (uint64_t x = 0; x < arg_stack.size(); x++) {
		if (_mem_size.find(arg_stack[x]) != _mem_size.end()){ 
			totalBytes += _mem_size[arg_stack[x]];
			if (_mem_addrs[arg_stack[x]].first > last_kernelexec) {
				readBytes += _mem_size[arg_stack[x]];
				loc.changed_read += _mem_size[arg_stack[x]];
			} else {
				read_unchanged += _mem_size[arg_stack[x]];
			}
			if (_mem_addrs[arg_stack[x]].second > last_kernelexec) {
				writeBytes += _mem_size[arg_stack[x]];
				loc.changed_write += _mem_size[arg_stack[x]];
			} else {
				write_unchanged += _mem_size[arg_stack[x]];
			}
		} 
	}
	loc.total_write = loc.changed_write + write_unchanged;
	loc.total_read = loc.changed_read + read_unchanged;


	_currentPhase->kernel_mem_info.push_back(loc);

	arg_stack.clear();
	_currentPhase->totalb += totalBytes;
	_currentPhase->readb += readBytes;
	_currentPhase->writeb += writeBytes;
	last_kernelexec = uni_counter;


}


void * PerfStorage::FindPtr(void * mem_loc) {
	for (std::map<void *, size_t >::iterator i = _mem_size.begin(); i != _mem_size.end(); ++i) {
		if ( mem_loc >= i->first && mem_loc < (void*)(((char *)i->first) + i->second)){
			return i->first;
		}
	}
	return (void *)NULL;
}

void PerfStorage::AddMemWrite(void * mem_loc){
	uni_counter++;

	if (_mem_size.find(mem_loc) == _mem_size.end()){
		void * trueMem = FindPtr(mem_loc);
		if (trueMem == NULL){
			fprintf(stderr, "%s: %p\n", "error finding WRITE mem location for", mem_loc);
			_mem_size[mem_loc] = 0; 
			_mem_addrs[mem_loc] = std::make_pair(uni_counter,uni_counter);
		} else {
			_mem_addrs[trueMem].second = uni_counter;	
		}

	} else {
		_mem_addrs[mem_loc].second = uni_counter;
	}
}


void PerfStorage::BeginPhase() {
	if (_currentPhase->phase_started == true) {
		EndPhase();
	}
	ZeroPhase(_currentPhase);
	gettimeofday(&begin_phase, NULL);
}

void PerfStorage::PushArgument(void * mem_loc){
	if (_mem_size.find(mem_loc) == _mem_size.end()) {
		void * trueMem = FindPtr(mem_loc);
		if (trueMem != NULL) {
			arg_stack.push_back(trueMem);
			return;
		}
	}
	arg_stack.push_back(mem_loc);
}

void PerfStorage::EndPhase() {
	gettimeofday(&end_phase, NULL);
	if (_currentPhase->phase_started == false) {
		ZeroPhase(_currentPhase);
		return;
	}

	CheckTimers(true);
	double avg_time = 0.0;
	if (_currentPhase->gpu_exe_count > 0) {
		avg_time=_currentPhase->gpu_exe_time / _currentPhase->gpu_exe_count;
	}
	char tmp[1500];
	snprintf(tmp, 1500, "phase_end,TotalTime:%lf,MallocTime:%lf,MallocSize:%llu,GPUTime:%lf,GPUAvg:%lf,MemcpyTime:%lf,MemcpySize:%llu,GTotal:%llu,GRead:%llu,GWrite:%llu",
			 diffTime(begin_phase, end_phase),
			 _currentPhase->mem_alloc_time,
			 _currentPhase->mem_alloc_size,
			 _currentPhase->gpu_exe_time,
			 avg_time,
			 _currentPhase->mem_transfer_time,
			 _currentPhase->mem_transfer_size,
			 _currentPhase->totalb,
			 _currentPhase->readb,
			 _currentPhase->writeb);
	LogEntry(tmp);
	ZeroPhase(_currentPhase);
}

void PerfStorage::WritePhasePart() {
	char tmp[1500];
	double avg_time = 0.0;
	if (_currentPhase->gpu_exe_count > 0) {
		avg_time=_currentPhase->gpu_exe_time / _currentPhase->gpu_exe_count;
	}
	snprintf(tmp, 1500, "in_progress,InProgressTotalTime:%lf,MallocTime:%lf,MallocSize:%llu,GPUTime:%lf,GPUAvg:%lf,MemcpyTime:%lf,MemcpySize:%llu,GTotal:%llu,GRead:%llu,GWrite:%llu",
			 diffTime(begin_phase, end_phase),
			 _currentPhase->mem_alloc_time,
			 _currentPhase->mem_alloc_size,
			 _currentPhase->gpu_exe_time,
			 avg_time,
			 _currentPhase->mem_transfer_time,
			 _currentPhase->mem_transfer_size,
			 _currentPhase->totalb,
			 _currentPhase->readb,
			 _currentPhase->writeb);
	LogEntry(tmp);	
}



PerfStorage::~PerfStorage(){
	EndPhase();
	outFile << _log.str();
	outFile.close();
}

/** Write an entry to the log */
void PerfStorage::LogEntry(char * entry) {
	struct timeval cur_time;
	gettimeofday(&cur_time, NULL);
	_log << entry << "\n";
	if (diffTime(last_write, cur_time) > 60){
		outFile << _log.str();
		outFile.flush();
		_log.str(std::string());
		last_write = cur_time;
	}
}



#endif
