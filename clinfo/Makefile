  OPENCL		:=$(CUDA_HOME)

  INC		:= -I$(OPENCL)/include
  LIB		:= -L$(OPENCL)/lib64 


all:	clinfo

clinfo:	clinfo.c
	gcc $(INC) $(LIB) -o clinfo clinfo.c -lOpenCL

clean:
	-rm clinfo
