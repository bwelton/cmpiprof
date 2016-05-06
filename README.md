# cmpiprof
MPI aware cuda profiler

The idea behind this profiler is to give an MPI context for CUDA exections. Specifically this is suppose to answer the question for each MPI phase (time between synchonous MPI calls) what is the GPU utilization within that phase.

Right now what is recorded is total time of the phase, total GPU kernel execution time in the phase, memory copy time/byte size, size of the memory region in bytes which are possibly reachable by GPU kernel executions in the phase, and the total amount of possibly modified data by the GPU during the phase. 

This is a work in progress and is a small part of a much larger effort. 

Configure options:

- Installation Directory setup: --prefix=<install prefix> 

- Includes needed for building CUDA applications: --cuda-include="-I<CUDA INCLUDES>.." (Not needed on Cray platforms)

- Link Options needed for building CUDA applications: --cuda-link="-L<Link Opts>..."  (Not needed on Cray Platforms)

- Boost Includes for the platform: --boost-include="-I<Boost Includes>..." (Not needed on Cray Platforms)

- Boost Libs for the platform: --boost-libs="-L<Boost Libs>..." (Not Needed on Cray Platforms)


Installation:

make && make install

Usage:

LD_PRELOAD="cuda_interceptor.so" ./Your_Application

Output:

A CSV file for each node (and each cuda process on said node) giving performance statistics for CUDA on an MPI phase basis.


