#pragma once
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

#ifndef __MAILBOX_H__
#define __MAILBOX_H__
#include "lk_globals.h"
#include "lk_mailbox.h"
#include "lk_utils.h"
#include <pthread.h>
/* Mailbox types */

typedef int mailbox_elem_t;
typedef mailbox_elem_t mailbox_t[MAX_NUM_BLOCKS];

mailbox_elem_t *d_to_device, *d_from_device, *h_to_device, *h_from_device;

#define lkHToDevice(_sm)        _vcast(h_to_device[_sm])
#define lkHFromDevice(_sm)      _vcast(h_from_device[_sm])
#define lkDToDevice(_sm)        _vcast(d_to_device[_sm])
#define lkDFromDevice(_sm)      _vcast(d_from_device[_sm])


pthread_t flushThread;
volatile char flushThread_run = 1, flushThread_go = 0;
void *
flushToDeviceAsync(void *fake)
{
  log("\n");
  while(flushThread_run)
  {
    if(flushThread_go)
    {
      checkCudaErrors(cudaMemcpyAsync(&d_to_device[0], &h_to_device[0], sizeof(mailbox_t), cudaMemcpyHostToDevice, backbone_stream));
      flushThread_go = 0;
    }
  }
  pthread_exit(NULL);
} // flushToDeviceAsync

ALWAYS_INLINE int
lkMailboxInit(cudaStream_t stream)
{
  log("\n");
  verb("sizeof(mailbox_elem_t) %lu sizeof(mailbox_t) %lu\n", sizeof(mailbox_elem_t), sizeof(mailbox_t));
  /* cudaHostAlloc: shared between host and GPU */
  checkCudaErrors(cudaHostAlloc((void **)&h_to_device, sizeof(mailbox_t), cudaHostAllocDefault));
  checkCudaErrors(cudaMalloc((void **)&d_to_device, sizeof(mailbox_t)));
  checkCudaErrors(cudaHostAlloc((void **)&h_from_device, sizeof(mailbox_t), cudaHostAllocDefault));
  checkCudaErrors(cudaMalloc((void **)&d_from_device, sizeof(mailbox_t)));
 
  verb("Created host-side mailbox @0x%x (TO) and 0x%x (FROM). Size is %lu.\n",
       _mycast_ &h_to_device[0], _mycast_ &h_from_device[0], sizeof(mailbox_t));
  verb("Created device-side mailbox @0x%x (TO) and 0x%x (FROM). Size is %lu.\n",
        _mycast_ &d_to_device[0], _mycast_ &d_from_device[0], sizeof(mailbox_t));
  
  int rc = pthread_create(&flushThread, NULL, flushToDeviceAsync, (void *) 0);
  if (rc)
    die("ERROR; return code from pthread_create() is %d\n", rc);
  return 0;
} // lkMailboxInit

ALWAYS_INLINE void
lkMailboxFree()
{
  flushThread_run = flushThread_go = 0;
  checkCudaErrors(cudaFree(d_from_device));
  checkCudaErrors(cudaFreeHost(h_from_device));
  checkCudaErrors(cudaFree(d_to_device));
  checkCudaErrors(cudaFreeHost(h_to_device));
} // lkMailboxFree

ALWAYS_INLINE void
lkMailboxPrint(const char *fn_name, int sm)
{
  log("[%s] to_device %s (%d), from_device %s (%d)\n", fn_name,
      getFlagName(lkHToDevice(sm)), lkHToDevice(sm), getFlagName(lkHFromDevice(sm)), lkHFromDevice(sm));
} // lkMailboxPrint


ALWAYS_INLINE void
lkMailboxFlush(bool to_device)
{
//   log("direction: %s\n", to_device ? "to_device": "from_device");
  if(to_device)
    checkCudaErrors(cudaMemcpyAsync(&d_to_device[0], &h_to_device[0], sizeof(mailbox_t), cudaMemcpyHostToDevice, backbone_stream));
  else
    checkCudaErrors(cudaMemcpyAsync(&h_from_device[0], &d_from_device[0], sizeof(mailbox_t), cudaMemcpyDeviceToHost, backbone_stream));
    
  cudaStreamSynchronize(backbone_stream);
} // lkMailboxFlush

ALWAYS_INLINE void
lkMailboxFlushAsync(bool to_device)
{
//   log("direction: %s\n", to_device ? "to_device": "from_device");
  if(to_device)
  {
#if 1
    if(!flushThread_go)
      flushThread_go = 1;
#else
    checkCudaErrors(cudaMemcpyAsync(&d_to_device[0], &h_to_device[0], sizeof(mailbox_t), cudaMemcpyHostToDevice, backbone_stream));
#endif
  }
  else
    checkCudaErrors(cudaMemcpyAsync(&h_from_device[0], &d_from_device[0], sizeof(mailbox_t), cudaMemcpyDeviceToHost, backbone_stream));
} // lkMailboxFlush

ALWAYS_INLINE void
lkMailboxFlushSMAsync(bool to_device, int sm)
{
  lkMailboxFlushAsync(to_device);
}

/* This simply doesn't work on D2H, probably because there is not much data to transfer...
 * Use lkMailboxFlush instead. Heavier, but safer.
 */
ALWAYS_INLINE void
lkMailboxFlushSM(bool to_device, int sm)
{
  lkMailboxFlush(to_device);
} // lkMailboxFlushSM

#endif /* __MAILBOX_H__ */
