#pragma once

#include <cstdio>

#include "lk_gpuMem.h"

__device__ void* devInputBufferPointers[NUM_EPOCHS];
__device__ void* devOutputBufferPointers[NUM_EPOCHS];


Arena input_arenas[NUM_EPOCHS];
Arena output_arenas[NUM_EPOCHS];
extern std::atomic<int> current_epoch = 0;

void init_arenas() {
	for(int i = 0; i < NUM_EPOCHS; i++) {
		input_arenas[i].reset();

		cudaMalloc(&input_arenas[i].base_ptr, INPUT_ARENA_SIZE);
		cudaMalloc(&output_arenas[i].base_ptr, OUTPUT_ARENA_SIZE);

		cudaMemcpyToSymbol(devInputBufferPointers[i],  &input_arenas[i].base_ptr, sizeof(input_arenas[i].base_ptr));
		cudaMemcpyToSymbol(devOutputBufferPointers[i],  &output_arenas[i].base_ptr, sizeof(output_arenas[i].base_ptr));
	}

	input_arenas[0].in_use.store(true);
	output_arenas[0].in_use.store(true);
}

void dealloc(int epoch){
	//numTasks is the previous number of active tasks, before dealloc of current task
	int numTasks = output_arenas[epoch].active_tasks.fetch_sub(1);
	input_arenas[epoch].active_tasks.fetch_sub(1);

	if(numTasks == 1) {
		output_arenas[epoch].reset();
		input_arenas[epoch].reset();
	}
}


int allocate(int input_bytes, int output_bytes) {
	if (input_bytes > INPUT_ARENA_SIZE){
		printf("input bytes over buffer size!\n");
		return -1;
	}
	if (output_bytes > OUTPUT_ARENA_SIZE) {
		printf("output bytes over buffer size!\n");
		return -1;
	}

	int current = current_epoch.load();

	if(input_arenas[current].offset.load() - input_bytes > INPUT_ARENA_SIZE || output_arenas[current].offset.load() - output_bytes > OUTPUT_ARENA_SIZE) {

		int next = (current + 1) % NUM_EPOCHS;
		if (input_arenas[next].in_use.load() == true) {
			return -1;	
		} 
		current_epoch.store(next);

		allocate(input_bytes, output_bytes);
		return 0;
	}

	input_arenas[current].active_tasks.fetch_add(1);
	output_arenas[current].active_tasks.fetch_add(1);

	input_arenas[current].offset.fetch_add(input_bytes);
	output_arenas[current].offset.fetch_add(output_bytes);
	return 1;

} 		

int getEpoch(){
	return current_epoch.load();
}
int getIOffset(int epoch){
	return input_arenas[epoch].offset.load();
}
int getROffset(int epoch){
	return output_arenas[epoch].offset.load();
}

