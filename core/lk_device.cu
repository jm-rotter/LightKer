/*
 *  LightKer - Light and flexible GPU persistent threads library
 *  Copyright (C) 2016  Paolo Burgio
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU Lesser General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU Lesser General Public License for more details.
 *
 *  You should have received a copy of the GNU Lesser General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
#pragma once
#include <stdio.h>

/* LK internal headers */
#include "lk_globals.h"
#include "lk_utils.h"
#include "lk_workqueue.h"
#include "lk_device.h"

/* App-specific data structures */
//#include "data.h"

#define DeviceWriteMyMailboxFrom(_val)  _vcast(from_device[blockIdx.x]) = (_val)
#define DeviceReadMyMailboxFrom()       _vcast(from_device[blockIdx.x])
#define DeviceReadMyMailboxTo()         _vcast(to_device[blockIdx.x])

/* App-specific functions: defined by user */
/* NOTE: formerly known as 'work_cuda' */
//__device__ int lkWorkCuda(volatile data_t *data, volatile res_t *res);
/* NOTE: formerly known as 'work_nocuda' */
//__device__ int lkWorkNoCuda(volatile data_t data, volatile res_t *res);

/* Main kernel function, for writing cuda aware work function.
 * Busy waits on the GPU until the CPU notifies new work, then
 * - t acknowledges the CPU and starts the real work. When finished
 * - it acknowledges the CPU through the trigger "from_device"
 * Formerly known as 'uniform_polling_cuda'
 */
__global__ void lkUniformPollingCuda(volatile mailbox_elem_t * to_device,
		volatile mailbox_elem_t * from_device)
{
	//printf("nothere\n");
	//__shared__ int res_shared;
	int blkid = blockIdx.x;
	int tid = threadIdx.x;

	//log("I am thread %d block %d, my SM ID is %d, my warp ID is %d, and my warp lane is %d\n",
	//		tid, blkid, __mysmid(), __mywarpid(), __mylaneid());
	//   log("data ptr @0x%x res ptr @0x%x lk_results @0x%x\n", _mycast_ data, _mycast_ res, _mycast_ lk_results);

	if(blkid != __mysmid())
	{
		/* Error! */
	}
	__syncthreads();

	if (tid == 0)
	{
		//     log("mailbox TO @ 0x%x FROM 0x%X\n", _mycast_ &to_device[blkid], _mycast_ &from_device[blkid]);
		//     log("Writing THREAD_NOP (%d) in from_device mailbox @0x%x\n", THREAD_NOP, (unsigned int) &from_device[blkid]);
		DeviceWriteMyMailboxFrom(THREAD_NOP);
		//res_shared = 0;
		//     log("Written THREAD_NOP (%d) in from_device mailbox @0x%x\n", from_device[blkid], (unsigned int) &from_device[blkid]);
	}  
	
	__syncthreads();
	int counter = 0;
	while (1)
	{
		// Shut down
		if (DeviceReadMyMailboxTo() == THREAD_EXIT) {
			break;
		}
		// Time to work!
		if (DeviceReadMyMailboxTo() == THREAD_WORK && DeviceReadMyMailboxFrom() != THREAD_FINISHED) {
			__syncthreads();
			dequeue(from_device);
			__syncthreads();
		}
		
		// Host got results
		else if (DeviceReadMyMailboxTo() == THREAD_NOP){
			DeviceWriteMyMailboxFrom(THREAD_NOP);
		}
//		else
//		{}

	} // while(1)

	__syncthreads();
	if(tid == 0)
		log("SM %d. Shutdown complete.\n", blkid);

} // lkUniformPollingCuda

/* Main kernel function, for writing "non-cuda" work function.
 * Busy waits on the GPU until the CPU notifies new work, then
 * - it acknowledges the CPU and starts the real work. When finished
 * - it acknowledges the CPU through the trigger "from_device"
 * Formerly known as 'uniform_polling'
 */
__global__ void lkUniformPollingNoCuda(volatile mailbox_elem_t * to_device,
		volatile mailbox_elem_t * from_device
		//                              volatile data_t *data, volatile res_t * res,
		) //                              lk_result_t *lk_results)
{
	LK_WARN_NOT_SUPPORTED();
}

__device__ uint32_t lkGetClusterID()
{
	if(blockIdx.x != __mysmid())
	{
		log("I am not on the right SM!");
	}
	return blockIdx.x;
}

__device__ uint32_t lkGetCoreID()
{
	return threadIdx.x;
}
