//
// auto-generated by op2.m on 30-May-2011 22:03:11
//

// user function

//#include "save_soln.h"

// host stub function

void op_par_loop_save_soln(char const *name, op_set set,
  op_arg arg0,
  op_arg arg1 ){
  
  cl_int ciErrNum;
  cl_event ceEvent;



  if (OP_diags>2) {
    printf(" kernel routine w/o indirection:  save_soln \n");
  }

  // initialise timers

  double cpu_t1, cpu_t2, wall_t1, wall_t2;
  op_timers(&cpu_t1, &wall_t1);

  // set CUDA execution parameters

  #ifdef OP_BLOCK_SIZE_0
    const size_t nthread = OP_BLOCK_SIZE_0;
  #else
    // int nthread = OP_block_size;
    const size_t nthread = 128;
  #endif

  const size_t nblocks = 200;
  const size_t n_tot_thread = nblocks * nthread;

  // work out shared memory requirements per element

  int nshared = 0;
  nshared = MAX(nshared,sizeof(float)*4);
  nshared = MAX(nshared,sizeof(float)*4);

  // execute plan

  int offset_s = nshared*OP_WARPSIZE;

  nshared = nshared*nthread;


  cl_kernel hKernel = getKernel( "op_cuda_save_soln" );

  //nshared *= 4;
  //offset_s *= 4;

  int i = 0;
  ciErrNum = clSetKernelArg( hKernel, i++, sizeof(cl_mem), &(arg0.data_d) );
  ciErrNum |= clSetKernelArg( hKernel, i++, sizeof(cl_mem), &(arg1.data_d) );
  ciErrNum |= clSetKernelArg( hKernel, i++, sizeof(int), &offset_s );
  ciErrNum |= clSetKernelArg( hKernel, i++, sizeof(int), &set->size );
  ciErrNum |= clSetKernelArg( hKernel, i++, nshared, NULL );
  assert_m( ciErrNum == CL_SUCCESS, "error setting kernel arguments" );

  ciErrNum = clEnqueueNDRangeKernel( cqCommandQueue, hKernel, 1, NULL, &n_tot_thread, &nthread, 0, NULL, &ceEvent );
  assert_m( ciErrNum == CL_SUCCESS, "error executing kernel" );

#ifndef ASYNC
  ciErrNum = clFinish( cqCommandQueue );
  assert_m( ciErrNum == CL_SUCCESS, "error completing device commands" );

#ifdef PROFILE
  unsigned long tqueue, tsubmit, tstart, tend, telapsed;
  ciErrNum  = clGetEventProfilingInfo( ceEvent, CL_PROFILING_COMMAND_QUEUED, sizeof(tqueue), &tqueue, NULL );
  ciErrNum |= clGetEventProfilingInfo( ceEvent, CL_PROFILING_COMMAND_SUBMIT, sizeof(tsubmit), &tsubmit, NULL );
  ciErrNum |= clGetEventProfilingInfo( ceEvent, CL_PROFILING_COMMAND_START, sizeof(tstart), &tstart, NULL );
  ciErrNum |= clGetEventProfilingInfo( ceEvent, CL_PROFILING_COMMAND_END, sizeof(tend), &tend, NULL );
  assert_m( ciErrNum == CL_SUCCESS, "error getting profiling info" );
  OP_kernels[0].queue_time      += (tsubmit - tqueue);
  OP_kernels[0].wait_time       += (tstart - tsubmit);
  OP_kernels[0].execution_time  += (tend - tstart);
  //printf("%20lu\n%20lu\n%20lu\n%20lu\n\n", tqueue, tsubmit, tstart, tend);
  //printf("queue: %8.4f\nwait:%8.4f\nexec: %8.4f\n\n", OP_kernels[0].queue_time * 1.0e-9, OP_kernels[0].wait_time * 1.0e-9, OP_kernels[0].execution_time * 1.0e-9 );
#endif

  // update kernel record

  op_timers(&cpu_t2, &wall_t2);
  op_timing_realloc(0);
  OP_kernels[0].name      = name;
  OP_kernels[0].count    += 1;
  OP_kernels[0].time     += wall_t2 - wall_t1;
  OP_kernels[0].transfer += (float)set->size * arg0.size;
  OP_kernels[0].transfer += (float)set->size * arg1.size;
#endif
}


