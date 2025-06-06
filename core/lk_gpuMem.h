#pragma once

#include <atomic>

#define NUM_EPOCHS 3
#define INPUT_ARENA_SIZE 512000000 //512 MB
#define OUTPUT_ARENA_SIZE 256000000

struct Arena {
	void* base_ptr;
	std::atomic<int> offset;
	std::atomic<int> active_tasks;
	std::atomic<int> in_use;

	void reset() {
		offset.store(0, std::memory_order_relaxed);
		active_tasks.store(0, std::memory_order_relaxed);
		in_use.store(false, std::memory_order_relaxed);
	}
};

extern Arena input_arenas[NUM_EPOCHS];
extern Arena output_arenas[NUM_EPOCHS];

extern __device__ void* devInputBufferPointers[NUM_EPOCHS];
extern __device__ void* devOutputBufferPointers[NUM_EPOCHS];


extern std::atomic<int> current_epoch;

void init_arenas();

void dealloc(int epoch);

int allocate(int input_bytes, int output_bytes);	

int getEpoch();
int getIOffset(int epoch);
int getROffset(int epoch);
