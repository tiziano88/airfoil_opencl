ifdef AMDAPPSDKROOT
  OPENCL = $(AMDAPPSDKROOT)
endif
ifdef CUDA_HOME
  OPENCL =$(CUDA_HOME)
endif

  INC = -I$(OPENCL)/include
  LIB = -L$(OPENCL)/lib64 


all:	clinfo

clinfo:	clinfo.c
	gcc $(INC) $(LIB) -o clinfo clinfo.c -lOpenCL

clean:
	-rm clinfo
