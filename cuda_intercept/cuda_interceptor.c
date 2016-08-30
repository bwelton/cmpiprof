/* Copyright Benjamin Welton 2015 */
#include "cuda_interceptor.h"
//#include "mpi.h"

#define __dv(v) = v

typedef size_t CUdeviceptr;
typedef size_t CUfunction;
typedef size_t CUstream;
double diffTime(timeval start, timeval end) {
	double t1=start.tv_sec+(start.tv_usec/1000000.0);
	double t2=end.tv_sec+(end.tv_usec/1000000.0);  
	return t2 - t1;
}

typedef __pid_t (*original_fork)(void);
__pid_t fork(void) {
	//fprintf(stderr,"we are inside fork\n");
	original_fork orig_cmal;
	orig_cmal = (original_fork)dlsym(RTLD_NEXT,"fork");	
	__pid_t ret = orig_cmal();
	return ret;
}

typedef int (*original_cudaDeviceReset)();
cudaError_t cudaDeviceReset() {
	original_cudaDeviceReset orig_cmal;	
	orig_cmal = (original_cudaDeviceReset)dlsym(RTLD_NEXT,"cudaDeviceReset");
	cudaError_t ret = (cudaError_t) orig_cmal();
	fprintf(stderr, "We called device reset\n");
	return ret;
}


typedef int (*original_cuMemAlloc_v2)(CUdeviceptr *dptr, unsigned int bytesize);
cudaError_t cuMemAlloc_v2(CUdeviceptr *dptr, unsigned int bytesize) {
	if (PerfStorageDataClass.get() == NULL) {
		fprintf(stderr, "%s\n", "Setting up our global data structure");
		PerfStorageDataClass.reset(new PerfStorage());
	}
	if (PerfStorageDataClass.get()->CheckThread() == false){
		fprintf(stderr, "%s\n", "GPU Thread Deleted, rebuilding");
		PerfStorageDataClass.get()->emergancyShutdown();
		PerfStorageDataClass.reset(new PerfStorage());
	}
	struct timeval time1, time2;
	original_cuMemAlloc_v2 orig_cmal;
	cudaEvent_t start;
	cudaEvent_t end;
	cudaEventCreate(&start);
	cudaEventCreate(&end);
	orig_cmal = (original_cuMemAlloc_v2)dlsym(RTLD_NEXT,"cuMemAlloc_v2");
	// Start Recording Timers
	cudaEventRecord(start);
	cudaError_t ret = (cudaError_t) orig_cmal(dptr, bytesize);
	cudaEventRecord(end);

	PerfStorageDataClass.get()->MallocTime(start, end, bytesize, (void *)*dptr);
	return ret;
}

typedef int (*original_cudaHostAlloc)(void ** pHost, size_t size, unsigned int flags);
cudaError_t cudaHostAlloc(void ** pHost, size_t size, unsigned int flags) {
	if (PerfStorageDataClass.get() == NULL) {
		fprintf(stderr, "%s\n", "Setting up our global data structure");
		PerfStorageDataClass.reset(new PerfStorage());
	}
	if (PerfStorageDataClass.get()->CheckThread() == false){
		fprintf(stderr, "%s\n", "GPU Thread Deleted, rebuilding");
		PerfStorageDataClass.get()->emergancyShutdown();
		PerfStorageDataClass.reset(new PerfStorage());
	}
	struct timeval time1, time2;
	original_cudaHostAlloc orig_cmal;
	cudaEvent_t start;
	cudaEvent_t end;
	cudaEventCreate(&start);
	cudaEventCreate(&end);


	orig_cmal = (original_cudaHostAlloc)dlsym(RTLD_NEXT,"cudaHostAlloc");
	// Start Recording Timers
	cudaEventRecord(start);
	cudaError_t ret = (cudaError_t) orig_cmal(pHost, size, flags);
	cudaEventRecord(end);

	PerfStorageDataClass.get()->MallocTime(start, end, size, (void *)*pHost);
	return ret;


}


typedef int (*original_cudaMalloc)(void ** devPtr, size_t size);
cudaError_t cudaMalloc(void **devPtr, size_t size) {
	if (PerfStorageDataClass.get() == NULL) {
		fprintf(stderr, "%s\n", "Setting up our global data structure");
		PerfStorageDataClass.reset(new PerfStorage());
	}
	if (PerfStorageDataClass.get()->CheckThread() == false){
		fprintf(stderr, "%s\n", "GPU Thread Deleted, rebuilding");
		PerfStorageDataClass.get()->emergancyShutdown();
		PerfStorageDataClass.reset(new PerfStorage());
	}
	struct timeval time1, time2;
	original_cudaMalloc orig_cmal;
	cudaEvent_t start;
	cudaEvent_t end;
	cudaEventCreate(&start);
	cudaEventCreate(&end);


	orig_cmal = (original_cudaMalloc)dlsym(RTLD_NEXT,"cudaMalloc");
	// Start Recording Timers
	cudaEventRecord(start);
	cudaError_t ret = (cudaError_t) orig_cmal(devPtr, size);
	cudaEventRecord(end);

	PerfStorageDataClass.get()->MallocTime(start, end, size, (void *)*devPtr);
	return ret;
}

extern "C" {

typedef int (*original_cudaCTXDestroy)(void * ctx);
cudaError_t cuCtxDestroy(void * ctx){
	original_cudaCTXDestroy orig_cmal;	
	orig_cmal = (original_cudaCTXDestroy)dlsym(RTLD_NEXT,"cuCtxDestroy");
	cudaError_t ret = (cudaError_t) orig_cmal(ctx);
	fprintf(stderr, "We called contextDestroy\n");
	return ret;
}



typedef int (*original_cuCtxCreate)(void *, void *, void *);
cudaError_t cuCtxCreate(void * a, void * b, void * c){
	original_cuCtxCreate orig_cmal;	
	orig_cmal = (original_cuCtxCreate)dlsym(RTLD_NEXT,"cuCtxCreate");
	cudaError_t ret = (cudaError_t) orig_cmal(a,b,c);
	fprintf(stderr, "We called cuCtxCreate\n");
	return ret;
}

typedef int (*original_cuCtxDetach)(void * ctx);
cudaError_t cuCtxDetach(void * ctx){
	original_cuCtxDetach orig_cmal;	
	orig_cmal = (original_cuCtxDetach)dlsym(RTLD_NEXT,"cuCtxDetach");
	cudaError_t ret = (cudaError_t) orig_cmal(ctx);
	fprintf(stderr, "We called CudaDetatch\n");
	return ret;
}
typedef int (*original_cudaThreadExit)();
cudaError_t cudaThreadExit(){
	original_cudaThreadExit orig_cmal;	
	orig_cmal = (original_cudaThreadExit)dlsym(RTLD_NEXT,"cudaThreadExit");
	// fprintf(stderr, "We called cudaThreadExit\n");
	PerfStorageDataClass.get()->CheckTimers(true);
	PerfStorageDataClass.get()->SetStream(0);
	cudaError_t ret = (cudaError_t) orig_cmal();
	return ret;
}

typedef int (*original_cuCtxPopCurrent)(void * ctx);
cudaError_t cuCtxPopCurrent(void * ctx){
	original_cuCtxPopCurrent orig_cmal;	
	orig_cmal = (original_cuCtxPopCurrent)dlsym(RTLD_NEXT,"cuCtxPopCurrent");
	cudaError_t ret = (cudaError_t) orig_cmal(ctx);
	fprintf(stderr, "We called PopCurrent\n");
	return ret;
} 

typedef int (*original_cuMemcpyHtoD_v2)(CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount);
cudaError_t cuMemcpyHtoD_v2(CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount) {
	if (PerfStorageDataClass.get() == NULL) {
		fprintf(stderr, "%s\n", "Setting up our global data structure");
		PerfStorageDataClass.reset(new PerfStorage());
	}
	if (PerfStorageDataClass.get()->CheckThread() == false){
		fprintf(stderr, "%s\n", "GPU Thread Deleted, rebuilding");
		PerfStorageDataClass.get()->emergancyShutdown();
		PerfStorageDataClass.reset(new PerfStorage());
	}
	struct timeval time1, time2;
	original_cuMemcpyHtoD_v2 orig_cmal;
	cudaEvent_t start;
	cudaEvent_t end;
	cudaEventCreate(&start);
	cudaEventCreate(&end);

	orig_cmal = (original_cuMemcpyHtoD_v2)dlsym(RTLD_NEXT,"cuMemcpyHtoD_v2");

	cudaEventRecord(start);
	cudaError_t ret = (cudaError_t) orig_cmal(dstDevice, srcHost, ByteCount);
	cudaEventRecord(end);

	PerfStorageDataClass.get()->AddMemoryTimer(start, end, ByteCount);

	PerfStorageDataClass.get()->AddMemWrite((void *)dstDevice);
	return ret;
}



typedef int (*original_cuMemcpyDtoH_v2)(void *dstHost, CUdeviceptr srcDevice, size_t ByteCount);
cudaError_t cuMemcpyDtoH_v2(void *dstHost, CUdeviceptr srcDevice, size_t ByteCount) {

	if (PerfStorageDataClass.get() == NULL) {
		fprintf(stderr, "%s\n", "Setting up our global data structure");
		PerfStorageDataClass.reset(new PerfStorage());
	}
	if (PerfStorageDataClass.get()->CheckThread() == false){
		fprintf(stderr, "%s\n", "GPU Thread Deleted, rebuilding");
		PerfStorageDataClass.get()->emergancyShutdown();
		PerfStorageDataClass.reset(new PerfStorage());
	}
	struct timeval time1, time2;
	original_cuMemcpyDtoH_v2 orig_cmal;
	cudaEvent_t start;
	cudaEvent_t end;
	cudaEventCreate(&start);
	cudaEventCreate(&end);

	orig_cmal = (original_cuMemcpyDtoH_v2)dlsym(RTLD_NEXT,"cuMemcpyDtoH_v2");

	cudaEventRecord(start);
	cudaError_t ret = (cudaError_t) orig_cmal(dstHost, srcDevice, ByteCount);
	cudaEventRecord(end);

	PerfStorageDataClass.get()->AddMemoryTimer(start, end, ByteCount);
	PerfStorageDataClass.get()->AddMemRead((void *)srcDevice);
	return ret;
}
typedef int (*original_createTexture)(cudaTextureObject_t *, const struct cudaResourceDesc *,
					 const struct cudaTextureDesc *, const struct cudaResourceViewDesc *);
cudaError_t cudaCreateTextureObject (cudaTextureObject_t * pTexObject,
           							 const struct cudaResourceDesc * pResDesc, 
           							 const struct cudaTextureDesc * pTexDesc, 
           							 const struct cudaResourceViewDesc * pResViewDesc) {

	struct timeval time1, time2;
	original_createTexture orig_cmal;
	cudaEvent_t start;
	cudaEvent_t end;

	orig_cmal = (original_createTexture)dlsym(RTLD_NEXT,"cudaCreateTextureObject");

	cudaError_t ret = (cudaError_t) orig_cmal(pTexObject, pResDesc, pTexDesc, pResViewDesc);

	return ret;
}

typedef int (*original_cudaMemcpy)(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind);
cudaError_t cudaMemcpy(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind) {

	if (PerfStorageDataClass.get() == NULL) {
		fprintf(stderr, "%s\n", "Setting up our global data structure");
		PerfStorageDataClass.reset(new PerfStorage());
	}
	if (PerfStorageDataClass.get()->CheckThread() == false){
		fprintf(stderr, "%s\n", "GPU Thread Deleted, rebuilding");
		PerfStorageDataClass.get()->emergancyShutdown();
		PerfStorageDataClass.reset(new PerfStorage());
	}
	struct timeval time1, time2;
	original_cudaMemcpy orig_cmal;
	cudaEvent_t start;
	cudaEvent_t end;
	cudaEventCreate(&start);
	cudaEventCreate(&end);

	orig_cmal = (original_cudaMemcpy)dlsym(RTLD_NEXT,"cudaMemcpy");

	cudaEventRecord(start);
	cudaError_t ret = (cudaError_t) orig_cmal(dst, src, count, kind);
	cudaEventRecord(end);

	PerfStorageDataClass.get()->AddMemoryTimer(start, end, count);
	if (kind == cudaMemcpyHostToDevice) {
		PerfStorageDataClass.get()->AddMemWrite(dst);
	} else if (kind == cudaMemcpyDeviceToHost) {
		PerfStorageDataClass.get()->AddMemRead((void *)src);
	}
	return ret;
}
}

typedef int (*original_cudaFree)(void *devPtr);
cudaError_t cudaFree(void *devPtr) {
	if (PerfStorageDataClass.get() == NULL) {
		fprintf(stderr, "%s\n", "Setting up our global data structure");
		PerfStorageDataClass.reset(new PerfStorage());
	}
	if (PerfStorageDataClass.get()->CheckThread() == false){
		fprintf(stderr, "%s\n", "GPU Thread Deleted, rebuilding");
		PerfStorageDataClass.get()->emergancyShutdown();
		PerfStorageDataClass.reset(new PerfStorage());
	}
	struct timeval time1, time2;
	original_cudaFree orig_cmal;

	orig_cmal = (original_cudaFree)dlsym(RTLD_NEXT,"cudaFree");

	gettimeofday(&time1, NULL);
	cudaError_t ret = (cudaError_t) orig_cmal(devPtr);
	gettimeofday(&time2, NULL);
	PerfStorageDataClass.get()->DeleteMem(devPtr);

	return ret;
}

/*** 
 * These are for intercepting the cuda launch commands
 * The order of these commands is the folllowing....
 * ... cudaConfigureCall <- Set the stream, grid dimensions, and block size for the call
 * ... cudaSetupArgument <- Push arguments onto the argument stack for execution
 * ... cudaLaunch <- Launch the kernel. 
 * ... In addition cudaLaunchKernel could be used.... Check this later
 ***/
typedef int (*original_cudaSetupArgument)(const void *arg, size_t size, size_t offset);
cudaError_t cudaSetupArgument(const void *arg, size_t size, size_t offset) {
	if (PerfStorageDataClass.get() == NULL) {
		fprintf(stderr, "%s\n", "Setting up our global data structure");
		PerfStorageDataClass.reset(new PerfStorage());
	}
	if (PerfStorageDataClass.get()->CheckThread() == false){
		fprintf(stderr, "%s\n", "GPU Thread Deleted, rebuilding");
		PerfStorageDataClass.get()->emergancyShutdown();
		PerfStorageDataClass.reset(new PerfStorage());
	}
	//struct timeval time1, time2;
	original_cudaSetupArgument orig_cmal;

	orig_cmal = (original_cudaSetupArgument)dlsym(RTLD_NEXT,"cudaSetupArgument");

	cudaError_t ret = (cudaError_t) orig_cmal(arg, size, offset);

	PerfStorageDataClass.get()->PushArgument((void *)*((size_t *)arg));

	return ret;	

}

typedef int (*original_cudaStreamDestroy)(cudaStream_t hStream);
cudaError_t cudaStreamDestroy(cudaStream_t hStream) {
	original_cudaStreamDestroy orig_stream;
	orig_stream = (original_cudaStreamDestroy)dlsym(RTLD_NEXT,"cudaStreamDestroy");
	PerfStorageDataClass.get()->CheckTimers(true);
	cudaError_t ret = (cudaError_t) orig_stream(hStream);
	PerfStorageDataClass.get()->SetStream(0);
	return ret;
}


typedef int (*original_cuStreamDestroy)(CUstream hStream);
cudaError_t cuStreamDestroy(CUstream hStream) {
	original_cuStreamDestroy orig_stream;
	orig_stream = (original_cuStreamDestroy)dlsym(RTLD_NEXT,"cuStreamDestroy");
	PerfStorageDataClass.get()->CheckTimers(true);
	cudaError_t ret = (cudaError_t) orig_stream(hStream);
	PerfStorageDataClass.get()->SetStream(0);
	return ret;
}

typedef int (*original_cudaConfigureCall)(dim3 gridDim, dim3 blockDim, size_t sharedMem, cudaStream_t stream);
cudaError_t cudaConfigureCall(dim3 gridDim, dim3 blockDim, size_t sharedMem, cudaStream_t stream ){
	if (PerfStorageDataClass.get() == NULL) {
		fprintf(stderr, "%s\n", "Setting up our global data structure");
		PerfStorageDataClass.reset(new PerfStorage());
	}
	if (PerfStorageDataClass.get()->CheckThread() == false){
		fprintf(stderr, "%s\n", "GPU Thread Deleted, rebuilding");
		PerfStorageDataClass.get()->emergancyShutdown();
		PerfStorageDataClass.reset(new PerfStorage());
	}
	original_cudaConfigureCall orig_cmal;

	orig_cmal = (original_cudaConfigureCall)dlsym(RTLD_NEXT,"cudaConfigureCall");

	cudaError_t ret = (cudaError_t) orig_cmal(gridDim, blockDim, sharedMem, stream);

	PerfStorageDataClass.get()->SetStream(stream);

	return ret;	

}
extern "C" {
typedef int (*original_cuLaunchKernel)(CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, 
									   unsigned int sharedMemBytes, CUstream hStream, void **kernelParams, void **extra);
cudaError_t cuLaunchKernel(CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ, unsigned int blockDimX, 
						   unsigned int blockDimY, unsigned int blockDimZ, 
						   unsigned int sharedMemBytes, CUstream hStream, void **kernelParams, void **extra) {
	if (PerfStorageDataClass.get() == NULL) {
		fprintf(stderr, "%s\n", "Setting up our global data structure");
		PerfStorageDataClass.reset(new PerfStorage());
	}
	if (PerfStorageDataClass.get()->CheckThread() == false){
		fprintf(stderr, "%s\n", "GPU Thread Deleted, rebuilding");
		PerfStorageDataClass.get()->emergancyShutdown();
		PerfStorageDataClass.reset(new PerfStorage());
	}
	original_cuLaunchKernel orig_cmal;
	cudaEvent_t start;
	cudaEvent_t end;
	cudaEventCreate(&start);
	cudaEventCreate(&end);


	orig_cmal = (original_cuLaunchKernel)dlsym(RTLD_NEXT,"cuLaunchKernel");

	cudaEventRecord(start);
	cudaError_t ret = (cudaError_t) orig_cmal(f,gridDimX,gridDimY,gridDimZ,blockDimX,blockDimY,blockDimZ,sharedMemBytes,hStream,kernelParams,extra);
	cudaEventRecord(end);

	PerfStorageDataClass.get()->AddGPUTimer(start,end);
	//PerfStorageDataClass.get()->LaunchedKernelParams();
	return ret;		
}
}

typedef int (*original_cudaLaunch)(const void *func);
cudaError_t cudaLaunch(const void *func) {
	if (PerfStorageDataClass.get() == NULL) {
		fprintf(stderr, "%s\n", "Setting up our global data structure");
		PerfStorageDataClass.reset(new PerfStorage());
	}
	if (PerfStorageDataClass.get()->CheckThread() == false){
		fprintf(stderr, "%s\n", "GPU Thread Deleted, rebuilding");
		PerfStorageDataClass.get()->emergancyShutdown();
		PerfStorageDataClass.reset(new PerfStorage());
	}
	original_cudaLaunch orig_cmal;
	cudaEvent_t start;
	cudaEvent_t end;
	cudaStream_t curStream = PerfStorageDataClass.get()->GetStream();
	cudaEventCreate(&start);
	cudaEventCreate(&end);


	orig_cmal = (original_cudaLaunch)dlsym(RTLD_NEXT,"cudaLaunch");

	cudaEventRecord(start,curStream);
	cudaError_t ret = (cudaError_t) orig_cmal(func);
	cudaEventRecord(end,curStream);

	PerfStorageDataClass.get()->AddGPUTimer(start,end);
	PerfStorageDataClass.get()->LaunchedKernel();


	return ret;		
}

typedef int (*original_cudaMemcpyAsync)(void *	dst, const void * src, size_t count, 
							 			enum cudaMemcpyKind kind, cudaStream_t stream);
cudaError_t cudaMemcpyAsync	(void *	dst, const void * src, size_t count, 
							 enum cudaMemcpyKind kind, cudaStream_t stream) {
	//fprintf(stderr, "%s\n", "In cuda cudaMemcpy async");
	if (PerfStorageDataClass.get() == NULL) {
		fprintf(stderr, "%s\n", "Setting up our global data structure");
		PerfStorageDataClass.reset(new PerfStorage());
	}
	if (PerfStorageDataClass.get()->CheckThread() == false){
		fprintf(stderr, "%s\n", "GPU Thread Deleted, rebuilding");
		PerfStorageDataClass.get()->emergancyShutdown();
		PerfStorageDataClass.reset(new PerfStorage());
	}
	original_cudaMemcpyAsync orig_cmal;
	cudaEvent_t start;
	cudaEvent_t end;
	cudaEventCreate(&start);
	cudaEventCreate(&end);

	orig_cmal = (original_cudaMemcpyAsync)dlsym(RTLD_NEXT,"cudaMemcpyAsync");

	cudaEventRecord(start);
	cudaError_t ret = (cudaError_t) orig_cmal(dst,src,count,kind,stream);
	cudaEventRecord(end);

	// Call our recorder
	PerfStorageDataClass.get()->AddMemoryTimer(start, end, count);
	if (kind == cudaMemcpyHostToDevice) {
		PerfStorageDataClass.get()->AddMemWrite(dst);
	} else if (kind == cudaMemcpyDeviceToHost) {
		PerfStorageDataClass.get()->AddMemRead((void *)src);
	}
	return ret;
}



extern "C" void __cudaRegisterFunction(void **fatCubinHandle, const char *hostFun, char *deviceFun,
    				       const char *deviceName, int thread_limit, uint3 *tid, uint3 *bid,
          			       dim3 *bDim, dim3 *gDim, int *wSize);


extern "C" {

// C/C++ MPI Functions
typedef int (*orig_Cwaitall)(int count, void * array_of_requests, void * array_of_statuses);
int MPI_Waitall(int count, void * array_of_requests, void * array_of_statuses) {

	if (PerfStorageDataClass.get() == NULL) {
		fprintf(stderr, "%s\n", "Setting up our global data structure");
		PerfStorageDataClass.reset(new PerfStorage());
	}
	if (PerfStorageDataClass.get()->CheckThread() == false){
		fprintf(stderr, "%s\n", "GPU Thread Deleted, rebuilding");
		PerfStorageDataClass.get()->emergancyShutdown();
		PerfStorageDataClass.reset(new PerfStorage());
	}
	PerfStorageDataClass.get()->EndPhase();

  	orig_Cwaitall orig_cmal;
  	orig_cmal = (orig_Cwaitall)dlsym(RTLD_NEXT,"MPI_Waitall");
  	int ret = orig_cmal(count, array_of_requests, array_of_statuses);
  	gettimeofday(&(PerfStorageDataClass.get()->begin_phase), NULL);
  	PerfStorageDataClass.get()->BeginPhase();
  	return ret;
}

typedef int (*orig_Cmpireduce)(const void *sendbuf, void *recvbuf, int count, int datatype,
               int op, int root, int comm);
int MPI_Reduce(const void *sendbuf, void *recvbuf, int count, int datatype,
               int op, int root, int comm) {
	if (PerfStorageDataClass.get() == NULL) {
		fprintf(stderr, "%s\n", "Setting up our global data structure");
		PerfStorageDataClass.reset(new PerfStorage());
	}
	if (PerfStorageDataClass.get()->CheckThread() == false){
		fprintf(stderr, "%s\n", "GPU Thread Deleted, rebuilding");
		PerfStorageDataClass.get()->emergancyShutdown();
		PerfStorageDataClass.reset(new PerfStorage());
	}

  	orig_Cmpireduce orig_cmal;
  	orig_cmal = (orig_Cmpireduce)dlsym(RTLD_NEXT,"MPI_Reduce");
  	return orig_cmal( sendbuf,  recvbuf,  count,  datatype,  op,  root,  comm);
}


typedef int (*orig_Cmpibar)(int comm);
int MPI_Barrier(int comm) {
	if (PerfStorageDataClass.get() == NULL) {
		fprintf(stderr, "%s\n", "Setting up our global data structure");
		PerfStorageDataClass.reset(new PerfStorage());
	}
	if (PerfStorageDataClass.get()->CheckThread() == false){
		fprintf(stderr, "%s\n", "GPU Thread Deleted, rebuilding");
		PerfStorageDataClass.get()->emergancyShutdown();
		PerfStorageDataClass.reset(new PerfStorage());
	}


	PerfStorageDataClass.get()->EndPhase();

  	orig_Cmpibar orig_cmal;
  	orig_cmal = (orig_Cmpibar)dlsym(RTLD_NEXT,"MPI_Barrier");
  	int ret = orig_cmal( comm);

    PerfStorageDataClass.get()->BeginPhase();
    return ret;
}

typedef int (*orig_Cmpiallreduce)(const void *sendbuf, void *recvbuf, int count, int datatype,
                  int op, int comm);
int MPI_Allreduce(const void *sendbuf, void *recvbuf, int count, int datatype,
                  int op, int comm) {

	if (PerfStorageDataClass.get() == NULL) {
		fprintf(stderr, "%s\n", "Setting up our global data structure");
		PerfStorageDataClass.reset(new PerfStorage());
	}
	if (PerfStorageDataClass.get()->CheckThread() == false){
		fprintf(stderr, "%s\n", "GPU Thread Deleted, rebuilding");
		PerfStorageDataClass.get()->emergancyShutdown();
		PerfStorageDataClass.reset(new PerfStorage());
	}
	PerfStorageDataClass.get()->EndPhase();
	// fprintf(stderr, "%s\n", "Inside MPIAllReduce");
  	orig_Cmpiallreduce orig_cmal;
  	orig_cmal = (orig_Cmpiallreduce)dlsym(RTLD_NEXT,"MPI_Allreduce");
  	int ret = orig_cmal( sendbuf,  recvbuf,  count,  datatype,  op,  comm);

  	PerfStorageDataClass.get()->BeginPhase();

  	return ret;
}


typedef int (*orig_Callgather)(const void *sendbuf, int sendcount, int sendtype, void *recvbuf,
                  int recvcount, int recvtype, int comm);

int MPI_Allgather(const void *sendbuf, int sendcount, int sendtype, void *recvbuf,
                  int recvcount, int recvtype, int comm) {
	if (PerfStorageDataClass.get() == NULL) {
		fprintf(stderr, "%s\n", "Setting up our global data structure");
		PerfStorageDataClass.reset(new PerfStorage());
	}
	if (PerfStorageDataClass.get()->CheckThread() == false){
		fprintf(stderr, "%s\n", "GPU Thread Deleted, rebuilding");
		PerfStorageDataClass.get()->emergancyShutdown();
		PerfStorageDataClass.reset(new PerfStorage());
	}

	PerfStorageDataClass.get()->EndPhase();

  	orig_Callgather orig_cmal;
  	orig_cmal = (orig_Callgather)dlsym(RTLD_NEXT,"MPI_Allgather");
  	int ret = orig_cmal(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm);

  	PerfStorageDataClass.get()->BeginPhase();

  	return ret;
}



// Fortran MPI functions
typedef void (*orig_waitall)(void * p1, void * p2, void * p3, void * ret);
void mpi_waitall_(void * p1, void * p2, void * p3, void * ret) {

	if (PerfStorageDataClass.get() == NULL) {
		fprintf(stderr, "%s\n", "Setting up our global data structure");
		PerfStorageDataClass.reset(new PerfStorage());
	}
	if (PerfStorageDataClass.get()->CheckThread() == false){
		fprintf(stderr, "%s\n", "GPU Thread Deleted, rebuilding");
		PerfStorageDataClass.get()->emergancyShutdown();
		PerfStorageDataClass.reset(new PerfStorage());
	}

	PerfStorageDataClass.get()->EndPhase();	


	// fprintf(stderr, "%s\n", "Inside WaitALL");
  	orig_waitall orig_cmal;
  	orig_cmal = (orig_waitall)dlsym(RTLD_NEXT,"mpi_waitall_");
  	orig_cmal(p1, p2, p3, ret);

  	PerfStorageDataClass.get()->BeginPhase();
}


typedef void (*orig_mpireduce)(void * p1, void * p2, void * p3, void * p4, 
		void * p5, void * p6, void * p7, void * p8);
void mpi_reduce_(void * p1, void * p2, void * p3, void * p4, void * p5, 
	void * p6, void * p7, void * p8) {
	if (PerfStorageDataClass.get() == NULL) {
		fprintf(stderr, "%s\n", "Setting up our global data structure");
		PerfStorageDataClass.reset(new PerfStorage());
	}
	if (PerfStorageDataClass.get()->CheckThread() == false){
		fprintf(stderr, "%s\n", "GPU Thread Deleted, rebuilding");
		PerfStorageDataClass.get()->emergancyShutdown();
		PerfStorageDataClass.reset(new PerfStorage());
	}

	// fprintf(stderr, "%s\n", "Inside MPIReduce");
  	orig_mpireduce orig_cmal;
  	orig_cmal = (orig_mpireduce)dlsym(RTLD_NEXT,"mpi_reduce_");
  	orig_cmal( p1,  p2,  p3,  p4,  p5,  p6,  p7, p8);
}

typedef void (*orig_mpibar)(void * p1, void * p2);
void mpi_barrier_(void * p1, void * p2) {
	if (PerfStorageDataClass.get() == NULL) {
		fprintf(stderr, "%s\n", "Setting up our global data structure");
		PerfStorageDataClass.reset(new PerfStorage());
	}
	if (PerfStorageDataClass.get()->CheckThread() == false){
		fprintf(stderr, "%s\n", "GPU Thread Deleted, rebuilding");
		PerfStorageDataClass.get()->emergancyShutdown();
		PerfStorageDataClass.reset(new PerfStorage());
	}

	PerfStorageDataClass.get()->EndPhase();	

	// fprintf(stderr, "%s\n", "Inside MPIBAR");
  	orig_mpibar orig_cmal;
  	orig_cmal = (orig_mpibar)dlsym(RTLD_NEXT,"mpi_barrier_");
  	orig_cmal( p1,  p2);

    PerfStorageDataClass.get()->BeginPhase();
}


typedef void (*orig_mpiallreduce)(void * p1, void * p2, void * p3, void * p4, 
		void * p5, void * p6, void * p7);
void mpi_allreduce_(void * p1, void * p2, void * p3, void * p4, void * p5, 
	void * p6, void * p7) {

	if (PerfStorageDataClass.get() == NULL) {
		fprintf(stderr, "%s\n", "Setting up our global data structure");
		PerfStorageDataClass.reset(new PerfStorage());
	}
	if (PerfStorageDataClass.get()->CheckThread() == false){
		fprintf(stderr, "%s\n", "GPU Thread Deleted, rebuilding");
		PerfStorageDataClass.get()->emergancyShutdown();
		PerfStorageDataClass.reset(new PerfStorage());
	}
	PerfStorageDataClass.get()->EndPhase();	
	// fprintf(stderr, "%s\n", "Inside MPIAllReduce");
  	orig_mpiallreduce orig_cmal;
  	orig_cmal = (orig_mpiallreduce)dlsym(RTLD_NEXT,"mpi_allreduce_");
  	orig_cmal( p1,  p2,  p3,  p4,  p5,  p6,  p7);

  	PerfStorageDataClass.get()->BeginPhase();
}

typedef void (*orig_charmBegin)(void * p1);
void charm_beginExecute (void * p1){
	if (PerfStorageDataClass.get() == NULL) {
		fprintf(stderr, "%s\n", "Setting up our global data structure");
		PerfStorageDataClass.reset(new PerfStorage());
	}
	if (PerfStorageDataClass.get()->CheckThread() == false){
		fprintf(stderr, "%s\n", "GPU Thread Deleted, rebuilding");
		PerfStorageDataClass.get()->emergancyShutdown();
		PerfStorageDataClass.reset(new PerfStorage());
	}
	PerfStorageDataClass.get()->EndPhase();		
  	orig_charmBegin orig_cmal;
  	orig_cmal = (orig_charmBegin)dlsym(RTLD_NEXT,"charm_beginExecute");
  	PerfStorageDataClass.get()->BeginPhase();
  	orig_cmal( p1);
}

typedef void (*orig_charm_beginComputation)();
void charm_beginComputation (){
	if (PerfStorageDataClass.get() == NULL) {
		fprintf(stderr, "%s\n", "Setting up our global data structure");
		PerfStorageDataClass.reset(new PerfStorage());
	}
	if (PerfStorageDataClass.get()->CheckThread() == false){
		fprintf(stderr, "%s\n", "GPU Thread Deleted, rebuilding");
		PerfStorageDataClass.get()->emergancyShutdown();
		PerfStorageDataClass.reset(new PerfStorage());
	}
	PerfStorageDataClass.get()->EndPhase();		
  	orig_charm_beginComputation orig_cmal;
  	orig_cmal = (orig_charm_beginComputation)dlsym(RTLD_NEXT,"charm_beginComputation");
  	PerfStorageDataClass.get()->BeginPhase();
  	orig_cmal();
}

typedef void (*orig_charm_beginPack)();
void charm_beginPack (){
	if (PerfStorageDataClass.get() == NULL) {
		fprintf(stderr, "%s\n", "Setting up our global data structure");
		PerfStorageDataClass.reset(new PerfStorage());
	}
	if (PerfStorageDataClass.get()->CheckThread() == false){
		fprintf(stderr, "%s\n", "GPU Thread Deleted, rebuilding");
		PerfStorageDataClass.get()->emergancyShutdown();
		PerfStorageDataClass.reset(new PerfStorage());
	}
	PerfStorageDataClass.get()->EndPhase();		
  	orig_charm_beginPack orig_cmal;
  	orig_cmal = (orig_charm_beginPack)dlsym(RTLD_NEXT,"charm_beginPack");
  	PerfStorageDataClass.get()->BeginPhase();
  	orig_cmal();
}

typedef void (*orig_charm_beginUnpack)();
void charm_beginUnpack (){
	if (PerfStorageDataClass.get() == NULL) {
		fprintf(stderr, "%s\n", "Setting up our global data structure");
		PerfStorageDataClass.reset(new PerfStorage());
	}
	if (PerfStorageDataClass.get()->CheckThread() == false){
		fprintf(stderr, "%s\n", "GPU Thread Deleted, rebuilding");
		PerfStorageDataClass.get()->emergancyShutdown();
		PerfStorageDataClass.reset(new PerfStorage());
	}
	PerfStorageDataClass.get()->EndPhase();		
  	orig_charm_beginUnpack orig_cmal;
  	orig_cmal = (orig_charm_beginUnpack)dlsym(RTLD_NEXT,"charm_beginUnpack");
  	PerfStorageDataClass.get()->BeginPhase();
  	orig_cmal();
}




typedef void (*orig_charmEnd)();
void charm_endExecute (){
	if (PerfStorageDataClass.get() == NULL) {
		fprintf(stderr, "%s\n", "Setting up our global data structure");
		PerfStorageDataClass.reset(new PerfStorage());
	}
	if (PerfStorageDataClass.get()->CheckThread() == false){
		fprintf(stderr, "%s\n", "GPU Thread Deleted, rebuilding");
		PerfStorageDataClass.get()->emergancyShutdown();
		PerfStorageDataClass.reset(new PerfStorage());
	}
	PerfStorageDataClass.get()->EndPhase();		
  	orig_charmEnd orig_cmal;
  	orig_cmal = (orig_charmEnd)dlsym(RTLD_NEXT,"charm_endExecute");
  	PerfStorageDataClass.get()->BeginPhase();
  	orig_cmal();
}

typedef void (*orig_charm_endComputation)();
void charm_endComputation (){
	if (PerfStorageDataClass.get() == NULL) {
		fprintf(stderr, "%s\n", "Setting up our global data structure");
		PerfStorageDataClass.reset(new PerfStorage());
	}
	if (PerfStorageDataClass.get()->CheckThread() == false){
		fprintf(stderr, "%s\n", "GPU Thread Deleted, rebuilding");
		PerfStorageDataClass.get()->emergancyShutdown();
		PerfStorageDataClass.reset(new PerfStorage());
	}
	PerfStorageDataClass.get()->EndPhase();		
  	orig_charm_endComputation orig_cmal;
  	orig_cmal = (orig_charm_endComputation)dlsym(RTLD_NEXT,"charm_endComputation");
  	PerfStorageDataClass.get()->BeginPhase();
  	orig_cmal();
}

typedef void (*orig_allgather)(double * value, int * sendcount, int * sendtype, double ** recvbuf,
                  int * recvcount, int * recvtype, int * comm, int * err);

void mpi_allgather_(double * value, int * sendcount, int * sendtype, double ** recvbuf,
                  int * recvcount, int * recvtype, int * comm, int * err) {
	if (PerfStorageDataClass.get() == NULL) {
		fprintf(stderr, "%s\n", "Setting up our global data structure");
		PerfStorageDataClass.reset(new PerfStorage());
	}
	if (PerfStorageDataClass.get()->CheckThread() == false){
		fprintf(stderr, "%s\n", "GPU Thread Deleted, rebuilding");
		PerfStorageDataClass.get()->emergancyShutdown();
		PerfStorageDataClass.reset(new PerfStorage());
	}

	PerfStorageDataClass.get()->EndPhase();	

  	orig_allgather orig_cmal;
  	orig_cmal = (orig_allgather)dlsym(RTLD_NEXT,"mpi_allgather_");
  	orig_cmal(value, sendcount, sendtype, recvbuf, recvcount, recvtype, comm, err);

  	PerfStorageDataClass.get()->BeginPhase();
}


 typedef void (*original_funcreg)(void **fatCubinHandle, const char *hostFun, char *deviceFun,
  							const char *deviceName, int thread_limit, uint3 *tid, uint3 *bid,
          					dim3 *bDim, dim3 *gDim, int *wSize);
 void __cudaRegisterFunction(void **fatCubinHandle, const char *hostFun, char *deviceFun,
    							const char *deviceName, int thread_limit, uint3 *tid, uint3 *bid,
          					dim3 *bDim, dim3 *gDim, int *wSize){
 	if (PerfStorageDataClass.get() == NULL) {
 		fprintf(stderr, "%s\n", "Setting up our global data structure");
 		PerfStorageDataClass.reset(new PerfStorage());
 	}
  	original_funcreg orig_cmal;
  	orig_cmal = (original_funcreg)dlsym(RTLD_NEXT,"__cudaRegisterFunction");
  	orig_cmal(fatCubinHandle, hostFun, deviceFun, deviceName, thread_limit, tid, bid, bDim, gDim, wSize);

 }
 }

