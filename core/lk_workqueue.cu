#include "utils.h"
#include "lk_globals.h"
#include "lk_mailbox.h"
#include "gpu_matmul.h"
#include "lk_utils.h"
#include "lk_workqueue.h"
#include "lk_host.h"
#include "utils.h"
#include <sys/types.h>



int h_queueHead; 
int h_taskCounter;

__device__ Input* d_input_queue;
__device__ int* d_tail;
__device__ int* d_taskCounter;
__device__ uint8_t* d_arg_buffer;

cudaStream_t qStream;
Input* diq;
int* dtc;
int* dt;
uint8_t* dab;

#define DeviceWriteMyMailboxFrom(_val)  _vcast(from_device[blockIdx.x]) = (_val)
#define HostWriteMyMailboxTo(_val)  _vcast(h_to_device[0]) = (_val)

void initQueue() {
	h_queueHead = 0;
	h_taskCounter = 0;

	//cudaMemset(&d_input_queue, 0, WORK_QUEUE_LENGTH * sizeof(Input));
	//cudaMemset(&d_tail, 0,  sizeof(int));
	//cudaMemcpyToSymbol(d_taskCounter, &h_taskCounter,  sizeof(int));
	//cudaMemset(&d_arg_buffer, 0, ARG_BUFFER_SIZE* sizeof(int));
	//

	cudaMalloc(&diq, sizeof(Input) * WORK_QUEUE_LENGTH);
	cudaMalloc(&dtc, sizeof(int));
	cudaMalloc(&dt, sizeof(int));
	cudaMalloc(&dab, sizeof(uint8_t) * ARG_BUFFER_SIZE);

	//init both to 0
	cudaMemcpy(dtc, &h_taskCounter, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dt, &h_taskCounter, sizeof(int), cudaMemcpyHostToDevice);
	
	cudaMemcpyToSymbol(d_input_queue, &diq, sizeof(Input*));
	cudaMemcpyToSymbol(d_taskCounter, &dtc, sizeof(int*));
	cudaMemcpyToSymbol(d_tail, &dt, sizeof(int*));
	cudaMemcpyToSymbol(d_arg_buffer, &dab, sizeof(dab));
	cudaStreamCreate(&qStream);

}


//__device__ const WorkFn lkTasks[] = {naive_wrapper, shared_wrapper};

const char* lkTasksDesc[] = {"naive", "shared_mem"};


__device__ void sleep() {
	return;
}

__device__ bool execute(Input* input) {
	log("Executing function 0\n");
	log("Offset is %d\n",input->offset);
	int offset = input->offset;
	naive_wrapper(offset);


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

	execute(&d_input_queue[idx]);

    DeviceWriteMyMailboxFrom(THREAD_FINISHED);
	return true;
}




bool enqueue(Input input, void* data, int size) {
	log("Enqueueing task: %s\n", lkTasksDesc[input.fn]);
	if(h_taskCounter >= WORK_QUEUE_LENGTH) {
		return false;
	}
	h_queueHead = (h_queueHead+1) % WORK_QUEUE_LENGTH;

	//printf("%d\n", ((int*)data)[0]);


	//uint8_t* pinned;
	//cudaMallocHost((void**)&pinned, size);
	//memcpy(pinned,data,size);

	//printf("%d\n", ((int*)pinned)[0]);

	cudaError_t err = cudaMemcpyAsync(dab, data, size, cudaMemcpyHostToDevice, qStream);

	printf("memcpy async error:%s\n", cudaGetErrorString(err));

	h_taskCounter++;

	err = cudaMemcpyAsync(diq, &input, sizeof(Input), cudaMemcpyHostToDevice, qStream);
		
	printf("memcpy async error:%s\n", cudaGetErrorString(err));

	cudaStreamSynchronize(qStream);
	
	log("memLoaded\n");
	HostWriteMyMailboxTo(THREAD_WORK);	
	lkMailboxPrint("Here", 0);

	lkMailboxFlushSM(false, 0);
	
	
	return true;
} 

