#pragma once
#define DeviceWriteMyMailboxFrom(_val)  _vcast(from_device[blockIdx.x]) = (_val)
#define DeviceReadMyMailboxFrom()       _vcast(from_device[blockIdx.x])
#define DeviceReadMyMailboxTo()         _vcast(to_device[blockIdx.x])


__global__ void lkUniformPollingCuda(volatile mailbox_elem_t * to_device, volatile mailbox_elem_t * from_device);


__global__ void lkUniformPollingNoCuda(volatile mailbox_elem_t * to_device, volatile mailbox_elem_t * from_device);


__device__ uint32_t lkGetClusterID();


__device__ uint32_t lkGetCoreID();
