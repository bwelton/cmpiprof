#include "cuda_interceptor.h"
thread_local std::shared_ptr<PerfStorage> PerfStorageDataClass;

void PerfStorage::AddHostMemPtrs(char * ptr) {
	if (_host_mem_ptrs.find(ptr) != _host_mem_ptrs.end())
		_host_mem_ptrs.insert(ptr);
}

void PerfStorage::CheckSend(char * ptr, char * source, size_t size) {
	if (_host_mem_ptrs.find(ptr) != _host_mem_ptrs.end()) {
		char tmp[1500];
		snprintf(tmp, 1500, "cuda_send,function:%s,size:%llu",source, size);
		LogEntry(tmp);
	}
}

void PerfStorage::StartTimer(int id) {
	struct timeval cur_time;
	gettimeofday(&cur_time, NULL);
	_runningStartTime[id] = cur_time;
}

void PerfStorage::EndTimer(int id){
	struct timeval cur_time;
	gettimeofday(&cur_time, NULL);
	if (_runningStartTime.find(id) != _runningStartTime.end()) {
		double tmp = diffTime(_runningStartTime[id],cur_time);
		if (_currentTime.find(id) != _currentTime.end())
			tmp = tmp + _currentTime[id];
		_currentTime[id] = tmp;
		_runningStartTime.erase(id);
	}
}

void PerfStorage::FinishTimer(int id, char * type) {
	char tmp[1500];
	if (_currentTime.find(id) != _currentTime.end()){
		snprintf(tmp, 1500, "reduction_timer,Type:%s,TotalTime:%f",type,_currentTime[id]);
		_currentTime.erase(id);
		LogEntry(tmp);	
	} else {
		fprintf(stderr, "%s\n", "WE COULD NOT FIND REDUCTION TIMER");
	}
}

void PerfStorage::BeginIdle(double curWallTime) {
	//if (_idle_timer.is_stopped() == true)
	//	_idle_timer.resume();
}

void PerfStorage::StopIdle(double curWallTime) {
	//if (_idle_timer.is_stopped() ==  false)
	//	_idle_timer.stop();
}

static void BeginTimers(void * value, double curWallTime) {
	//if (PerfStorageDataClass.get() == NULL) {
	//	fprintf(stderr, "%s\n", "Setting up our global data structure - Begin");
	//	PerfStorageDataClass.reset(new PerfStorage());
	//}
	//PerfStorageDataClass.get()->BeginIdle(curWallTime);
}

static void EndTimers(void * value, double curWallTime) {
	if (PerfStorageDataClass.get() == NULL) {
		fprintf(stderr, "%s\n", "Setting up our global data structure - End");
		PerfStorageDataClass.reset(new PerfStorage());
	}
	//PerfStorageDataClass.get()->StopIdle(curWallTime);
}

void PerfStorage::RegisterCharmCallbacks() {
	callbacksRegistered = 1;
	// CcdCallOnConditionKeep(CcdPROCESSOR_BEGIN_IDLE,&(BeginTimers),0);
	// CcdCallOnConditionKeep(CcdPROCESSOR_END_IDLE,&(EndTimers),0);	
	//CcdCallOnConditionKeep(CcdPROCESSOR_BEGIN_BUSY,&(EndTimers),this);
}

int PerfStorage::GetCharmPhase(){
	return charm_phase;
}

void PerfStorage::SetCharmPhase(int phase) {
	charm_phase = phase;
}


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
	_host_mem_ptrs.clear();
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
		cudaEventSynchronize(t[x].second);
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
	//fprintf(stderr,"%p\n",mem_loc);
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
	if (diffTime(last_write, cur_time) > 5){
		outFile << _log.str();
		outFile.flush();
		_log.str(std::string());
		last_write = cur_time;
	}
}