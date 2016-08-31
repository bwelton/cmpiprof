#include <stdio.h>
#include <assert.h>

inline
cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %sn", 
            cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
#endif
  return result;
}

int main(void)
{
  float tmp;
  size_t dataSizes[] = {100000 / 4,1000000 / 4,5000000 / 4,10000000 / 4,20000000 / 4,40000000 / 4,80000000 / 4,100000000/4};
  int count = 8;
  printf("Size of float: %u\n", sizeof(float));
  for (int i = 0; i < count; i++) {
  	float mallocTime, copyToDevice, copyFromDevice;


  	float * x = (float *) malloc(dataSizes[i]*sizeof(float));
  	memset(x, 1, dataSizes[i]*sizeof(float));
  	float * d_x;
  	{
	  	cudaEvent_t start;
	  	cudaEvent_t end;
	  	cudaEventCreate(&start);
		cudaEventCreate(&end);

		cudaEventRecord(start);
		checkCuda(cudaMalloc(&d_x, dataSizes[i]*sizeof(float)));
		cudaEventRecord(end);
		cudaEventSynchronize(end);
		cudaEventElapsedTime(&mallocTime, start,end);
		cudaEventDestroy(start);
		cudaEventDestroy(end);
	}
	{
	  	cudaEvent_t start;
	  	cudaEvent_t end;
	  	cudaEventCreate(&start);
		cudaEventCreate(&end);

		cudaEventRecord(start);
		checkCuda(cudaMemcpy(d_x, x, dataSizes[i]*sizeof(float), cudaMemcpyHostToDevice));
		cudaEventRecord(end);
		cudaEventSynchronize(end);
		cudaEventElapsedTime(&copyToDevice, start,end);
		cudaEventDestroy(start);
		cudaEventDestroy(end);		
	} 
	{
	  	cudaEvent_t start;
	  	cudaEvent_t end;
	  	cudaEventCreate(&start);
		cudaEventCreate(&end);

		cudaEventRecord(start);
		checkCuda(cudaMemcpy(x,d_x, dataSizes[i]*sizeof(float), cudaMemcpyDeviceToHost));
		cudaEventRecord(end);
		cudaEventSynchronize(end);
		cudaEventElapsedTime(&copyFromDevice, start,end);
		cudaEventDestroy(start);
		cudaEventDestroy(end);	
	}
	cudaFree(d_x);
	free(x);
	printf("%llu,%f,%f,%f\n",dataSizes[i] * 4,mallocTime/1000,copyToDevice/1000,copyFromDevice/1000);
  }
}