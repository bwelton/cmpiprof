#include <stdio.h>
extern "C" {
void INTER_PhaseStart(void) {
	// Do nothing here, We are going to get intercepted
	fprintf(stderr,"We shouldn't be here\n");
}

void CMPIPROF_StartTimer(int identifier, char * name) {
	fprintf(stderr, "%s\n", "ERROR - IN SHIM START TIMER" );
}

void CMPIPROF_EndTimer(int identifier, char * name) {
	fprintf(stderr, "%s\n", "ERROR - IN SHIM END TIMER");
}

void CMPIPROF_FinishTimer(int identifier, char * name) {
	fprintf(stderr, "%s\n", "ERROR - IN SHIM FINISH TIMER");	
}


}
