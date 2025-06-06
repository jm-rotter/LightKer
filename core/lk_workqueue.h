#pragma once
#include "lk_mailbox.h"
#define WORK_QUEUE_LENGTH 32
//To get around different signatures for different function pointers
//typedef void (*WorkFn)(void* arg, void* res);

typedef struct Task{
	int fn;
	int input_offset;
	int output_offset;
	int epoch;
} Task;

void initQueue();

extern __device__ Task* d_task_queue;

__device__ bool dequeue(volatile mailbox_elem_t * from_device);

int enqueue(void* data, int input_size, int output_size, int taskId);

extern Task* dtq;
extern int* dt;
extern cudaStream_t qStream;
