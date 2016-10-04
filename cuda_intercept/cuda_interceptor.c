/* Copyright Benjamin Welton 2016 */
#include "cuda_interceptor.h"

#define CUDA_MEMORY_TIMERS(FUNC) \
    cudaEvent_t start; \
    cudaEvent_t end; \
    cudaEventCreate(&start); \
    cudaEventCreate(&end); \
    cudaEventRecord(start); \
    FUNC \
    cudaEventRecord(end); \

#define CUDA_TIME_STREAM(FUNC) \
    cudaEvent_t start; \
    cudaEvent_t end; \
    cudaEventCreate(&start); \
    cudaEventCreate(&end); \
    cudaStream_t curStream = PerfStorageDataClass.get()->GetStream(); \
    cudaEventRecord(start,curStream); \
    FUNC \
    cudaEventRecord(end,curStream);


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

extern "C" {

// CUDA Memory Allocators
typedef int (*original_cuMemHostAlloc)(void ** pp,size_t bytesize, unsigned int Flags);
cudaError_t cuMemHostAlloc(void ** pp,size_t bytesize, unsigned int Flags)  {
    BUILD_STORAGE_CLASS

    original_cuMemHostAlloc orig_cmal;
    orig_cmal = (original_cuMemHostAlloc)dlsym(RTLD_NEXT,"cuMemHostAlloc");

    CUDA_MEMORY_TIMERS(cudaError_t ret = (cudaError_t) orig_cmal(pp, bytesize, Flags);)

    PerfStorageDataClass.get()->MallocTime(start, end, bytesize, (void *)*pp);
    return ret;
}   

typedef int (*original_cuMemAlloc_v2)(CUdeviceptr *dptr, unsigned int bytesize);
cudaError_t cuMemAlloc_v2(CUdeviceptr *dptr, unsigned int bytesize) {
    BUILD_STORAGE_CLASS

    original_cuMemAlloc_v2 orig_cmal;
    orig_cmal = (original_cuMemAlloc_v2)dlsym(RTLD_NEXT,"cuMemAlloc_v2");

    CUDA_MEMORY_TIMERS(cudaError_t ret = (cudaError_t) orig_cmal(dptr, bytesize);)

    PerfStorageDataClass.get()->MallocTime(start, end, bytesize, (void *)*dptr);
    return ret;
}

typedef int (*original_cudaHostAlloc)(void ** pHost, size_t size, unsigned int flags);
cudaError_t cudaHostAlloc(void ** pHost, size_t size, unsigned int flags) {
    BUILD_STORAGE_CLASS

    original_cudaHostAlloc orig_cmal;
    orig_cmal = (original_cudaHostAlloc)dlsym(RTLD_NEXT,"cudaHostAlloc");
    
    CUDA_MEMORY_TIMERS(cudaError_t ret = (cudaError_t) orig_cmal(pHost, size, flags);)
    
    PerfStorageDataClass.get()->MallocTime(start, end, size, (void *)*pHost);
    return ret;
}

typedef int (*original_cuTexRefSetAddress_v2)(size_t * pHost,void * a2, CUdeviceptr size, unsigned int flags);
cudaError_t cuTexRefSetAddress_v2 (size_t *a1, void * a2, CUdeviceptr a3, size_t a4) {
    BUILD_STORAGE_CLASS

    original_cuTexRefSetAddress_v2 orig_cmal;
    orig_cmal = (original_cuTexRefSetAddress_v2)dlsym(RTLD_NEXT,"cuTexRefSetAddress_v2");

    CUDA_MEMORY_TIMERS(cudaError_t ret = (cudaError_t) orig_cmal(a1, a2, a3,a4);)
    
    PerfStorageDataClass.get()->MallocTime(start, end, a4, (void *)a3);
    return ret;
}


typedef int (*original_cudaMalloc)(void ** devPtr, size_t size);
cudaError_t cudaMalloc(void **devPtr, size_t size) {
    BUILD_STORAGE_CLASS

    original_cudaMalloc orig_cmal;
    orig_cmal = (original_cudaMalloc)dlsym(RTLD_NEXT,"cudaMalloc");

    CUDA_MEMORY_TIMERS(cudaError_t ret = (cudaError_t) orig_cmal(devPtr, size);)

    PerfStorageDataClass.get()->MallocTime(start, end, size, (void *)*devPtr);
    return ret;
}

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
typedef int (*original_cudaMemcpyAsync)(void *  dst, const void * src, size_t count, 
                                        enum cudaMemcpyKind kind, cudaStream_t stream);
cudaError_t cudaMemcpyAsync (void * dst, const void * src, size_t count, 
                             enum cudaMemcpyKind kind, cudaStream_t stream) {
    BUILD_STORAGE_CLASS

    original_cudaMemcpyAsync orig_cmal;
    orig_cmal = (original_cudaMemcpyAsync)dlsym(RTLD_NEXT,"cudaMemcpyAsync");   

    CUDA_MEMORY_TIMERS(cudaError_t ret = (cudaError_t) orig_cmal(dst,src,count,kind,stream);)

    // Call our recorder
    PerfStorageDataClass.get()->AddMemoryTimer(start, end, count);
    if (kind == cudaMemcpyHostToDevice) {
        PerfStorageDataClass.get()->AddMemWrite(dst);
    } else if (kind == cudaMemcpyDeviceToHost) {
        PerfStorageDataClass.get()->AddHostMemPtrs((char*)dst);
        PerfStorageDataClass.get()->AddMemRead((void *)src);
    }

    return ret;
}


typedef int (*original_cuMemcpyHtoD_v2)(CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount);
cudaError_t cuMemcpyHtoD_v2(CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount) {
    BUILD_STORAGE_CLASS

    original_cuMemcpyHtoD_v2 orig_cmal;
    orig_cmal = (original_cuMemcpyHtoD_v2)dlsym(RTLD_NEXT,"cuMemcpyHtoD_v2");
    
    CUDA_MEMORY_TIMERS(cudaError_t ret = (cudaError_t) orig_cmal(dstDevice, srcHost, ByteCount);)

    PerfStorageDataClass.get()->AddMemoryTimer(start, end, ByteCount);

    PerfStorageDataClass.get()->AddMemWrite((void *)dstDevice);
    return ret;
}

typedef int (*original_cuMemcpyDtoH_v2)(void *dstHost, CUdeviceptr srcDevice, size_t ByteCount);
cudaError_t cuMemcpyDtoH_v2(void *dstHost, CUdeviceptr srcDevice, size_t ByteCount) {

    BUILD_STORAGE_CLASS

    original_cuMemcpyDtoH_v2 orig_cmal;
    orig_cmal = (original_cuMemcpyDtoH_v2)dlsym(RTLD_NEXT,"cuMemcpyDtoH_v2");

    CUDA_MEMORY_TIMERS(cudaError_t ret = (cudaError_t) orig_cmal(dstHost, srcDevice, ByteCount);)

    PerfStorageDataClass.get()->AddMemoryTimer(start, end, ByteCount);
    PerfStorageDataClass.get()->AddHostMemPtrs((char*)dstHost);
    PerfStorageDataClass.get()->AddMemRead((void *)srcDevice);
    return ret;
}

typedef int (*original_cudaMemcpy)(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind);
cudaError_t cudaMemcpy(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind) {

    BUILD_STORAGE_CLASS

    original_cudaMemcpy orig_cmal;
    orig_cmal = (original_cudaMemcpy)dlsym(RTLD_NEXT,"cudaMemcpy");
    
    CUDA_MEMORY_TIMERS(cudaError_t ret = (cudaError_t) orig_cmal(dst, src, count, kind);)

    PerfStorageDataClass.get()->AddMemoryTimer(start, end, count);
    if (kind == cudaMemcpyHostToDevice) {
        PerfStorageDataClass.get()->AddMemWrite(dst);
    } else if (kind == cudaMemcpyDeviceToHost) {
        PerfStorageDataClass.get()->AddHostMemPtrs((char*)dst);
        PerfStorageDataClass.get()->AddMemRead((void *)src);
    }
    return ret;
}

typedef int (*original_cudaFree)(void *devPtr);
cudaError_t cudaFree(void *devPtr) {
    BUILD_STORAGE_CLASS

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
    BUILD_STORAGE_CLASS

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
    BUILD_STORAGE_CLASS

    if (PerfStorageDataClass.get()->callbacksRegistered == 0) {
        PerfStorageDataClass.get()->RegisterCharmCallbacks();
    }
    original_cudaConfigureCall orig_cmal;

    orig_cmal = (original_cudaConfigureCall)dlsym(RTLD_NEXT,"cudaConfigureCall");

    cudaError_t ret = (cudaError_t) orig_cmal(gridDim, blockDim, sharedMem, stream);

    PerfStorageDataClass.get()->SetStream(stream);

    return ret; 

}


typedef int (*original_cuLaunchKernel)(CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, 
                                       unsigned int sharedMemBytes, CUstream hStream, void **kernelParams, void **extra);
cudaError_t cuLaunchKernel(CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ, unsigned int blockDimX, 
                           unsigned int blockDimY, unsigned int blockDimZ, 
                           unsigned int sharedMemBytes, CUstream hStream, void **kernelParams, void **extra) {
    BUILD_STORAGE_CLASS

    // if (PerfStorageDataClass.get()->callbacksRegistered == 0) {
    //  PerfStorageDataClass.get()->RegisterCharmCallbacks();
    // }

    original_cuLaunchKernel orig_cmal;
    orig_cmal = (original_cuLaunchKernel)dlsym(RTLD_NEXT,"cuLaunchKernel");

    // Handle locally listed kernel parameters
    if (kernelParams != NULL) {
        char *** tmp_cast = (char***)(kernelParams);
        int skip = 0;
        while (tmp_cast[skip] != NULL){
            PerfStorageDataClass.get()->PushArgument((void *)**(tmp_cast+skip));
            skip++;
        }
    }
    if (extra != NULL)
        fprintf(stderr, "%s\n", "We also have extra parameters");

    CUDA_MEMORY_TIMERS(cudaError_t ret = (cudaError_t) orig_cmal(f,gridDimX,gridDimY,gridDimZ,blockDimX,blockDimY,blockDimZ,sharedMemBytes,hStream,kernelParams,extra);)

    PerfStorageDataClass.get()->AddGPUTimer(start,end);
    PerfStorageDataClass.get()->LaunchedKernel();
    return ret;     
}


typedef int (*original_cudaLaunch)(const void *func);
cudaError_t cudaLaunch(const void *func) {
    BUILD_STORAGE_CLASS

    if (PerfStorageDataClass.get()->callbacksRegistered == 0) {
        PerfStorageDataClass.get()->RegisterCharmCallbacks();
    }
    original_cudaLaunch orig_cmal;
    orig_cmal = (original_cudaLaunch)dlsym(RTLD_NEXT,"cudaLaunch");
    
    CUDA_TIME_STREAM(cudaError_t ret = (cudaError_t) orig_cmal(func);)

    PerfStorageDataClass.get()->AddGPUTimer(start,end);
    PerfStorageDataClass.get()->LaunchedKernel();

    return ret;     
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
extern "C" void __cudaRegisterFunction(void **fatCubinHandle, const char *hostFun, char *deviceFun,
                       const char *deviceName, int thread_limit, uint3 *tid, uint3 *bid,
                       dim3 *bDim, dim3 *gDim, int *wSize);
}

// typedef void (*orig_QDStatebcastQD1)( void ** state, void ** msg);

// void _Z13_handlePhase0P7QdStateP5QdMsg( void ** state, void ** msg) {
//  if (PerfStorageDataClass.get() == NULL) {
//      fprintf(stderr, "%s\n", "Setting up our global data structure");
//      PerfStorageDataClass.reset(new PerfStorage());
//  }
//  if (PerfStorageDataClass.get()->CheckThread() == false){
//      fprintf(stderr, "%s\n", "GPU Thread Deleted, rebuilding");
//      PerfStorageDataClass.get()->emergancyShutdown();
//      PerfStorageDataClass.reset(new PerfStorage());
//  }

//  fprintf(stderr,"ending phase charm - HP0\n");
// //       fprintf(stderr,"My Values %p,%d,%d\n",t,flag,count);
//  PerfStorageDataClass.get()->EndPhase();     
//  PerfStorageDataClass.get()->BeginPhase();
    
//  //PerfStorageDataClass.get()->SetCharmPhase(flag);

    
//  orig_QDStatebcastQD1 orig_cmal = (orig_QDStatebcastQD1)dlsym(RTLD_NEXT,"_Z13_handlePhase0P7QdStateP5QdMsg");
    
//  orig_cmal(state, msg);  
// }
// //typedef void (*orig_QDStatebcastQD1)( void ** state, void ** msg);

// void _Z13_handlePhase1P7QdStateP5QdMsg( void ** state, void ** msg) {
//  if (PerfStorageDataClass.get() == NULL) {
//      fprintf(stderr, "%s\n", "Setting up our global data structure");
//      PerfStorageDataClass.reset(new PerfStorage());
//  }
//  if (PerfStorageDataClass.get()->CheckThread() == false){
//      fprintf(stderr, "%s\n", "GPU Thread Deleted, rebuilding");
//      PerfStorageDataClass.get()->emergancyShutdown();
//      PerfStorageDataClass.reset(new PerfStorage());
//  }

//  fprintf(stderr,"ending phase charm - HP1\n");
// //       fprintf(stderr,"My Values %p,%d,%d\n",t,flag,count);
//  PerfStorageDataClass.get()->EndPhase();     
//  PerfStorageDataClass.get()->BeginPhase();
    
//  //PerfStorageDataClass.get()->SetCharmPhase(flag);

    
//  orig_QDStatebcastQD1 orig_cmal = (orig_QDStatebcastQD1)dlsym(RTLD_NEXT,"_Z13_handlePhase1P7QdStateP5QdMsg");
    
//  orig_cmal(state, msg);  
// }
// //typedef void (*orig_QDStatebcastQD1)( void ** state, void ** msg);

// void _Z13_handlePhase2P7QdStateP5QdMsg( void ** state, void ** msg) {
//  if (PerfStorageDataClass.get() == NULL) {
//      fprintf(stderr, "%s\n", "Setting up our global data structure");
//      PerfStorageDataClass.reset(new PerfStorage());
//  }
//  if (PerfStorageDataClass.get()->CheckThread() == false){
//      fprintf(stderr, "%s\n", "GPU Thread Deleted, rebuilding");
//      PerfStorageDataClass.get()->emergancyShutdown();
//      PerfStorageDataClass.reset(new PerfStorage());
//  }

//  fprintf(stderr,"ending phase charm -HP2\n");
// //       fprintf(stderr,"My Values %p,%d,%d\n",t,flag,count);
//  PerfStorageDataClass.get()->EndPhase();     
//  PerfStorageDataClass.get()->BeginPhase();
    
//  //PerfStorageDataClass.get()->SetCharmPhase(flag);

    
//  orig_QDStatebcastQD1 orig_cmal = (orig_QDStatebcastQD1)dlsym(RTLD_NEXT,"_Z13_handlePhase2P7QdStateP5QdMsg");
    
//  orig_cmal(state, msg);  
// }
// //typedef void (*orig_QDStatebcastQD1)( void ** state, void ** msg);

// void _Z9_bcastQD2P7QdStateP5QdMsg( void ** state, void ** msg) {
//  if (PerfStorageDataClass.get() == NULL) {
//      fprintf(stderr, "%s\n", "Setting up our global data structure");
//      PerfStorageDataClass.reset(new PerfStorage());
//  }
//  if (PerfStorageDataClass.get()->CheckThread() == false){
//      fprintf(stderr, "%s\n", "GPU Thread Deleted, rebuilding");
//      PerfStorageDataClass.get()->emergancyShutdown();
//      PerfStorageDataClass.reset(new PerfStorage());
//  }

//  fprintf(stderr,"ending phase charm\n");
// //       fprintf(stderr,"My Values %p,%d,%d\n",t,flag,count);
//  PerfStorageDataClass.get()->EndPhase();     
//  PerfStorageDataClass.get()->BeginPhase();
    
//  //PerfStorageDataClass.get()->SetCharmPhase(flag);

    
//  orig_QDStatebcastQD1 orig_cmal = (orig_QDStatebcastQD1)dlsym(RTLD_NEXT,"_Z9_bcastQD2P7QdStateP5QdMsg");
    
//  orig_cmal(state, msg);  
// }

// typedef void (*orig_QDStateSendCount)(void ** t, int flag, int count);

// void _ZN7QdState9sendCountEii(void ** t, int flag, int count) {
//  if (PerfStorageDataClass.get() == NULL) {
//      fprintf(stderr, "%s\n", "Setting up our global data structure");
//      PerfStorageDataClass.reset(new PerfStorage());
//  }
//  if (PerfStorageDataClass.get()->CheckThread() == false){
//      fprintf(stderr, "%s\n", "GPU Thread Deleted, rebuilding");
//      PerfStorageDataClass.get()->emergancyShutdown();
//      PerfStorageDataClass.reset(new PerfStorage());
//  }
//  if (flag > 1 && PerfStorageDataClass.get()->GetCharmPhase() != flag){
//      fprintf(stderr,"ending phase charm\n");
//      fprintf(stderr,"My Values %p,%d,%d\n",t,flag,count);
//      PerfStorageDataClass.get()->EndPhase();     
//      PerfStorageDataClass.get()->BeginPhase();
//  }
//  PerfStorageDataClass.get()->SetCharmPhase(flag);

    
//  orig_QDStateSendCount orig_cmal = (orig_QDStateSendCount)dlsym(RTLD_NEXT,"_ZN7QdState9sendCountEii");
    
//  orig_cmal(t, flag, count);  
// }


// typedef int (*original_createTexture)(cudaTextureObject_t *, const struct cudaResourceDesc *,
//                   const struct cudaTextureDesc *, const struct cudaResourceViewDesc *);
// cudaError_t cudaCreateTextureObject (cudaTextureObject_t * pTexObject,
//                                       const struct cudaResourceDesc * pResDesc, 
//                                       const struct cudaTextureDesc * pTexDesc, 
//                                       const struct cudaResourceViewDesc * pResViewDesc) {

//  original_createTexture orig_cmal;
//  orig_cmal = (original_createTexture)dlsym(RTLD_NEXT,"cudaCreateTextureObject");
//  cudaError_t ret = (cudaError_t) orig_cmal(pTexObject, pResDesc, pTexDesc, pResViewDesc);
//  return ret;
// }
