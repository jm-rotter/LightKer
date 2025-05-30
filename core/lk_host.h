#pragma once
#include "lk_mailbox.h"

void *syncMalboxFrom(void * fake);


extern mailbox_elem_t *d_to_device, *d_from_device, *h_to_device, *h_from_device;



int lkNumThreadsPerSM();

int lkNumClusters();

void lkLaunch(void (*kernel) (volatile mailbox_elem_t *, volatile mailbox_elem_t *),
              dim3 blknum, dim3 blkdim, int shmem);


void lkInit(unsigned int blknum_x, unsigned int blkdim_x,
						int shmem, bool cudaMode);

void lkTriggerSM(int sm);

void lkTriggerMultiple();

void lkDispose();
void lkHostSleep(long time_ns);
