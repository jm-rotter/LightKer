#include "utils.h"
#include "lk_globals.h"
#include "lk_mailbox.h"
#include "gpu_matmul.h"
#include "lk_utils.h"
#include "lk_workqueue.h"

Task h_queue[WORK_QUEUE_LENGTH];
int h_queueHead, h_taskCounter = 0;

__device__ Task *d_queue;
__device__ int *d_tail, *d_taskCounter;

#define DeviceWriteMyMailboxFrom(_val)  _vcast(from_device[blockIdx.x]) = (_val)

void initQueue() {
	h_queueHead = h_taskCounter = 0;
	cudaMalloc(&d_queue, WORK_QUEUE_LENGTH * sizeof(Task));
	cudaMalloc(&d_tail, sizeof(int));
	cudaMalloc(&d_taskCounter, sizeof(int));

	int zero = 0;

	cudaMemcpy(d_queue, h_queue, WORK_QUEUE_LENGTH * sizeof(Task), cudaMemcpyHostToDevice);
	cudaMemcpy(d_tail, &zero, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_taskCounter, &h_taskCounter, sizeof(int), cudaMemcpyHostToDevice);
}


//__device__ const WorkFn lkTasks[] = {naive_wrapper, shared_wrapper};

const char* lkTasksDesc[] = {"naive", "shared_mem"};


__device__ void sleep() {
	return;
}

__device__ bool execute(Task* task) {
	//lkTasks[task->input.fn](task->input.arg, task->res);
	naive_wrapper(task->input.arg, task->res);
	return true;
}

__device__ bool dequeue(volatile mailbox_elem_t * from_device){
	int count = atomicSub(d_taskCounter, 1);

	if(count <= 0) {
		atomicAdd(d_taskCounter, 1);
		sleep();
		return false;
	}

	int tail = atomicAdd(d_tail, 1);
	int idx = tail % WORK_QUEUE_LENGTH;

	execute(&d_queue[idx]);

    DeviceWriteMyMailboxFrom(THREAD_FINISHED);
	return true;
}




bool enqueue(Task task) {
	log("Enqueueing task: %s", lkTasksDesc[task.input.fn]);
	if(h_taskCounter >= WORK_QUEUE_LENGTH) {
		return false;
	}

	int idx = h_queueHead % WORK_QUEUE_LENGTH;
	h_queue[idx] = task;

	cudaMemcpy(&(d_queue + idx)->input, &task.input, sizeof(Input), cudaMemcpyHostToDevice);

	h_taskCounter++;
	h_queueHead++;
	
	cudaMemcpy(d_taskCounter, &h_taskCounter, sizeof(int), cudaMemcpyHostToDevice);

	//HostWriteMyMailboxTo(THREAD_WORK);	

	return true;
} 
