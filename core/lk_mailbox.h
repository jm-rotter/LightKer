#pragma once
#include "lk_globals.h"
typedef int mailbox_elem_t;
typedef mailbox_elem_t mailbox_t[MAX_NUM_BLOCKS];

#define lkHToDevice(_sm)        _vcast(h_to_device[_sm])
#define lkHFromDevice(_sm)      _vcast(h_from_device[_sm])
#define lkDToDevice(_sm)        _vcast(d_to_device[_sm])
#define lkDFromDevice(_sm)      _vcast(d_from_device[_sm])


extern cudaStream_t backbone_stream;


 int
lkMailboxInit(cudaStream_t stream = 0);


 void
lkMailboxFree();

 void
lkMailboxPrint(const char *fn_name, int sm);

 void
lkMailboxFlush(bool to_device);

 void
lkMailboxFlushAsync(bool to_device);

 void
lkMailboxFlushSMAsync(bool to_device, int sm);

 void
lkMailboxFlushSM(bool to_device, int sm);
