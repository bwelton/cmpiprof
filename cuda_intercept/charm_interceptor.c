/* Copyright Benjamin Welton 2016 */
#include "cuda_interceptor.h"

extern "C" {
void CMPIPROF_StartTimer(int identifier, char * name) {
	BUILD_STORAGE_CLASS

	PerfStorageDataClass.get()->StartTimer(identifier);	
}

void CMPIPROF_EndTimer(int identifier, char * name) {
	BUILD_STORAGE_CLASS

	PerfStorageDataClass.get()->EndTimer(identifier);	
}

void CMPIPROF_FinishTimer(int identifier, char * name) {
	BUILD_STORAGE_CLASS

	PerfStorageDataClass.get()->FinishTimer(identifier,name);	
}

typedef void (*orig_charmBegin)(void * p1);
void charm_beginExecute (void * p1){
	BUILD_STORAGE_CLASS

	PerfStorageDataClass.get()->EndPhase();		
  	orig_charmBegin orig_cmal;
  	orig_cmal = (orig_charmBegin)dlsym(RTLD_NEXT,"charm_beginExecute");
  	PerfStorageDataClass.get()->BeginPhase();
  	orig_cmal( p1);
}

void INTER_PhaseStart(void) {
	BUILD_STORAGE_CLASS

	//fprintf(stderr, "%s\n", "Init Phase Called");
	PerfStorageDataClass.get()->EndPhase();	
  	PerfStorageDataClass.get()->BeginPhase();
}

typedef void (*orig_charm_beginComputation)();
void charm_beginComputation (){
	BUILD_STORAGE_CLASS

	PerfStorageDataClass.get()->EndPhase();		
  	orig_charm_beginComputation orig_cmal;
  	orig_cmal = (orig_charm_beginComputation)dlsym(RTLD_NEXT,"charm_beginComputation");
  	PerfStorageDataClass.get()->BeginPhase();
  	orig_cmal();
}

typedef void (*orig_charm_beginPack)();
void charm_beginPack (){
	BUILD_STORAGE_CLASS

	PerfStorageDataClass.get()->EndPhase();		
  	orig_charm_beginPack orig_cmal;
  	orig_cmal = (orig_charm_beginPack)dlsym(RTLD_NEXT,"charm_beginPack");
  	PerfStorageDataClass.get()->BeginPhase();
  	orig_cmal();
}

typedef void (*orig_charm_beginUnpack)();
void charm_beginUnpack (){
	BUILD_STORAGE_CLASS

	PerfStorageDataClass.get()->EndPhase();		
  	orig_charm_beginUnpack orig_cmal;
  	orig_cmal = (orig_charm_beginUnpack)dlsym(RTLD_NEXT,"charm_beginUnpack");
  	PerfStorageDataClass.get()->BeginPhase();
  	orig_cmal();
}

typedef void (*orig_charmEnd)();
void charm_endExecute (){
	BUILD_STORAGE_CLASS

	PerfStorageDataClass.get()->EndPhase();		
  	orig_charmEnd orig_cmal;
  	orig_cmal = (orig_charmEnd)dlsym(RTLD_NEXT,"charm_endExecute");
  	PerfStorageDataClass.get()->BeginPhase();
  	orig_cmal();
}

typedef void (*orig_charm_endComputation)();
void charm_endComputation (){
	BUILD_STORAGE_CLASS

	PerfStorageDataClass.get()->EndPhase();		
  	orig_charm_endComputation orig_cmal;
  	orig_cmal = (orig_charm_endComputation)dlsym(RTLD_NEXT,"charm_endComputation");
  	PerfStorageDataClass.get()->BeginPhase();
  	orig_cmal();
}
}