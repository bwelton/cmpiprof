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

#define BUILD_STORAGE_CLASS \
	if (PerfStorageDataClass.get() == NULL) { \
		fprintf(stderr, "%s\n", "Setting up our global data structure"); \
		PerfStorageDataClass.reset(new PerfStorage()); \
	} \
	if (PerfStorageDataClass.get()->CheckThread() == false){ \
		fprintf(stderr, "%s\n", "GPU Thread Deleted, rebuilding"); \
		PerfStorageDataClass.get()->emergancyShutdown(); \
		PerfStorageDataClass.reset(new PerfStorage()); \
	}


//#include <boost/timer/timer.hpp>

#define CcdPROCESSOR_BEGIN_BUSY 0
#define CcdPROCESSOR_END_IDLE 0 /*Synonym*/
#define CcdPROCESSOR_BEGIN_IDLE 1
#define CcdPROCESSOR_END_BUSY 1 /*Synonym*/
#define CcdPROCESSOR_STILL_IDLE 2

extern "C" {
typedef void (*CcdVoidFn)(void *userParam,double curWallTime);
int CcdCallOnConditionKeep(int condnum, CcdVoidFn fnp, void *arg);
}


double diffTime(timeval start, timeval end);
extern "C" {
void INTER_PhaseStart(void);
}

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
		//_idle_timer.start();
		//_idle_timer.stop();
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
		charm_phase = 0;
		arg_stack.clear();
		callbacksRegistered = 0;
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
	void SetCharmPhase(int phase);
	int GetCharmPhase();
	void BeginIdle(double curWallTime);
	void StopIdle(double curWallTime);
	void RegisterCharmCallbacks();
	void StartTimer(int id);
	void EndTimer(int id);
	void FinishTimer(int id, char * type);
	void AddHostMemPtrs(char * ptr);
	void CheckSend(char * ptr, char * source, size_t size);
	int callbacksRegistered;
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
	int charm_phase;
	uint64_t uni_counter;
	uint64_t last_kernelexec;
	std::map<void *, std::pair<uint64_t,uint64_t> > _mem_addrs;
	std::map<void *, size_t > _mem_size;
	std::vector<void *> arg_stack;
	double _time_waiting;
	std::map<int, double> _currentTime; 
	std::map<int, struct timeval> _runningStartTime;

	std::set<char*> _host_mem_ptrs;

	//boost::timer::cpu_timer _idle_timer;
};

extern thread_local std::shared_ptr<PerfStorage> PerfStorageDataClass;
#endif
