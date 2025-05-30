#pragma once
#include "lk_mailbox.h"
#define WORK_QUEUE_LENGTH 32

//To get around different signatures for different function pointers
typedef void (*WorkFn)(void* arg, void* res);

typedef struct Input{
	int fn;
	int offset;
} Input;



typedef struct Output {
	int offset;
} Output;


void initQueue();


__device__ bool dequeue(volatile mailbox_elem_t * from_device);

bool enqueue(Input input);
