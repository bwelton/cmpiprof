/* Copyright Benjamin Welton 2016 */
#include "cuda_interceptor.h"
extern "C" {
// C/C++ MPI Functions
typedef int (*orig_Cwaitall)(int count, void * array_of_requests, void * array_of_statuses);
int MPI_Waitall(int count, void * array_of_requests, void * array_of_statuses) {

	BUILD_STORAGE_CLASS

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
	BUILD_STORAGE_CLASS
    PerfStorageDataClass.get()->CheckSend((char*)sendbuf, "REDUCE", size_t(count));

  	orig_Cmpireduce orig_cmal;
  	orig_cmal = (orig_Cmpireduce)dlsym(RTLD_NEXT,"MPI_Reduce");
  	return orig_cmal( sendbuf,  recvbuf,  count,  datatype,  op,  root,  comm);
}


typedef int (*orig_Cmpibar)(int comm);
int MPI_Barrier(int comm) {
	BUILD_STORAGE_CLASS



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

	BUILD_STORAGE_CLASS
  PerfStorageDataClass.get()->CheckSend((char*)sendbuf, "ALL_REDUCE", size_t(count));

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
	BUILD_STORAGE_CLASS

  PerfStorageDataClass.get()->CheckSend((char*)sendbuf, "ALL_GATHER", size_t(sendcount));
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

	BUILD_STORAGE_CLASS


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
	BUILD_STORAGE_CLASS

  
	// fprintf(stderr, "%s\n", "Inside MPIReduce");
  	orig_mpireduce orig_cmal;
  	orig_cmal = (orig_mpireduce)dlsym(RTLD_NEXT,"mpi_reduce_");
  	orig_cmal( p1,  p2,  p3,  p4,  p5,  p6,  p7, p8);
}

typedef void (*orig_mpibar)(void * p1, void * p2);
void mpi_barrier_(void * p1, void * p2) {
	BUILD_STORAGE_CLASS


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

	BUILD_STORAGE_CLASS

	PerfStorageDataClass.get()->EndPhase();	
	// fprintf(stderr, "%s\n", "Inside MPIAllReduce");
  	orig_mpiallreduce orig_cmal;
  	orig_cmal = (orig_mpiallreduce)dlsym(RTLD_NEXT,"mpi_allreduce_");
  	orig_cmal( p1,  p2,  p3,  p4,  p5,  p6,  p7);

  	PerfStorageDataClass.get()->BeginPhase();
}

typedef void (*orig_allgather)(double * value, int * sendcount, int * sendtype, double ** recvbuf,
                  int * recvcount, int * recvtype, int * comm, int * err);

void mpi_allgather_(double * value, int * sendcount, int * sendtype, double ** recvbuf,
                  int * recvcount, int * recvtype, int * comm, int * err) {
  BUILD_STORAGE_CLASS


  PerfStorageDataClass.get()->EndPhase(); 

    orig_allgather orig_cmal;
    orig_cmal = (orig_allgather)dlsym(RTLD_NEXT,"mpi_allgather_");
    orig_cmal(value, sendcount, sendtype, recvbuf, recvcount, recvtype, comm, err);

    PerfStorageDataClass.get()->BeginPhase();
}
}