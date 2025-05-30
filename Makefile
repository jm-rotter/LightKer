APPDIR = ${LK_HOME}/workQueueTest/matrix-mul-bench/
EXENAME = persistent
APPSRCS = $(APPDIR)/src/gpu_matmul.cu

# PARAMS += -DL_MAX_LENGTH=20
# PARAMS += -DWORK_TIME=200000
PARAMS += ${PAR1} ${PAR2}

include ${LK_HOME}/Makefile.lk
