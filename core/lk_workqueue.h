#pragma once
#include "lk_mailbox.h"
#define WORK_QUEUE_LENGTH 32
#define ARG_BUFFER_SIZE    1024*1024*1024
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

extern __device__ uint8_t* d_arg_buffer;
extern __device__ Input* d_input_queue;

__device__ bool dequeue(volatile mailbox_elem_t * from_device);

bool enqueue(Input input, void* data, int size);
extern Input* diq;
extern int* dtc;
extern int* dt;
extern uint8_t* dab;
extern cudaStream_t qStream;
