#ifndef _UTILITY_H_
#define _UTILITY_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

typedef struct InputArgs
{
  long arg;
  char marker[64];
} Args;

void parseArgs(Args* aptr, int* argc, char** argv)
{
  if (*argc < 3)
  {
    fprintf(stderr, "[Error] To few arguments!\nUsage:\n%s [a] [marker]\n", argv[0]);
    exit(EXIT_FAILURE);
  }
  else
  {
    aptr->arg = atoll(argv[1]);
    memset((aptr->marker), 0, 64);
    strncpy((aptr->marker), argv[2], 63);
  }
  *argc = 1;
  return;
}

void ins__printtime(struct timeval *start, struct timeval *stop, char *marker) {

  long time=1000000*(stop->tv_sec-start->tv_sec)+stop->tv_usec-start->tv_usec;
  printf("\n%s: Execution time = %ld microseconds\n", marker, time);

  return;
}

long get_time(struct timeval *start, struct timeval *stop) {
  return 1000000*(stop->tv_sec-start->tv_sec)+stop->tv_usec-start->tv_usec;
}

#endif