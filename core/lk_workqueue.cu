#include "utils.h"
#include "lk_globals.h"
#include "lk_mailbox.h"
#include "gpu_matmul.h"
#include "lk_utils.h"
#include "lk_workqueue.h"
#include "lk_host.h"
#include "utils.h"
#include "lk_gpuMem.h"
#include <sys/types.h>



int h_queueHead; 
int h_taskCounter;

__device__ Task* d_task_queue;
__device__ int* d_tail;

cudaStream_t qStream;
Task* dtq; //Host pointer to device task queue 
int* dt;  //Host pointer to alloc tail

#define DeviceWriteMyMailboxFrom(_val)  _vcast(from_device[blockIdx.x]) = (_val)
#define HostWriteMyMailboxTo(_val)  _vcast(h_to_device[0]) = (_val)

void initQueue() {
	h_queueHead = 0;
	h_taskCounter = 0;

	cudaMalloc(&dtq, sizeof(Task) * WORK_QUEUE_LENGTH);
	cudaMalloc(&dt, sizeof(int));

	//Just init tail to 0. As h_taskcounter is already 0
	cudaMemcpy(dt, &h_taskCounter, sizeof(int), cudaMemcpyHostToDevice);
	
	cudaMemcpyToSymbol(d_task_queue, &dtq, sizeof(Task*));
	cudaMemcpyToSymbol(d_tail, &dt, sizeof(int*));
	cudaStreamCreate(&qStream);
}

const char* lkTasksDesc[] = {"naive", "shared_mem"};

__device__ void sleep() {
	return;
}

__device__ bool execute(Task task) {
	log("Executing function 0\n");
	log("Input Offset is %d\n",task.input_offset);
	log("Output Offset is %d\n",task.output_offset);

	naive_wrapper(task);

	//lkTasks[task->input.fn](task->input.arg, task->res);
	//naive_wrapper(task->input.arg, task->res);
	return true;
}

__device__ bool dequeue(volatile mailbox_elem_t * from_device){

	log("%s\n", "in dequeue");
	//log("%d\n", &d_taskCounter);

	//int count = atomicSub(d_taskCounter, 1);
	int count = 1;
	log("%d\n", count);
	if(count <= 0) {
		//atomicAdd(d_taskCounter, 1);
		sleep();
		return false;
	}

	//int tail = atomicAdd(d_tail, 1);
	int tail = 0;
	int idx = tail % WORK_QUEUE_LENGTH;

	execute(d_task_queue[idx]);

    DeviceWriteMyMailboxFrom(THREAD_FINISHED);
	return true;
}



//returns task idx in task queue
int enqueue(void* data, int input_size, int output_size, int taskId) {
	log("Enqueueing task: %d\n", taskId);
	if(h_taskCounter >= WORK_QUEUE_LENGTH) {
		return -1;
	}

	Task task;
	task.fn = taskId;
	
	int epoch = getEpoch();
	int ioffset = getIOffset(epoch);
	int roffset = getROffset(epoch);

	int success = allocate(input_size, output_size);

	switch(success) {
		case -1:
			return -1;
			break;
		case 0:
			task.epoch = (epoch + 1) % NUM_EPOCHS;
			task.input_offset = 0;
			task.output_offset = 0;
			break;
		case 1: 
			task.epoch = epoch;
			task.input_offset = ioffset;
			task.output_offset = roffset;
			break;
	}

	cudaError_t err = cudaMemcpyAsync(input_arenas[task.epoch].base_ptr + task.input_offset, data, input_size, cudaMemcpyHostToDevice, qStream);

	printf("Memcpy task data async error:%s\n", cudaGetErrorString(err));
	

	err = cudaMemcpyAsync(dtq + h_queueHead, &task, sizeof(Task), cudaMemcpyHostToDevice, qStream);

	printf("Memcpy task metadata async error:%s\n", cudaGetErrorString(err));
	
	h_queueHead = (h_queueHead+1) % WORK_QUEUE_LENGTH;
	
	h_taskCounter++;

	//Going to remove at some point Host shouldn't need to notify GPU of new tasks
	cudaStreamSynchronize(qStream);
	
	log("memLoaded\n");
	HostWriteMyMailboxTo(THREAD_WORK);	
	lkMailboxPrint("Here", 0);

	lkMailboxFlushSM(false, 0);
	
	return h_queueHead -1;
} 

