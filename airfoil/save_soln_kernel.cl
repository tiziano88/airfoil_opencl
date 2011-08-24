#define OP_WARPSIZE 32
#define MIN(a,b) ((a<b) ? (a) : (b))
inline void save_soln(float *q, float *qold){
 for (int n=0; n<4; n++) qold[n] = q[n];
}
__kernel void op_cuda_save_soln(
  __global float *arg0,
  __global float *arg1,
  int   offset_s,
  int   set_size ) {

  float arg0_l[4];
  float arg1_l[4];
  int   tid = get_local_id(0) % OP_WARPSIZE;

  //extern  " //__local
  __local char shared[6400];" //64000
  __local float *arg_s = (__local float *) (shared + offset_s*(get_local_id(0)/OP_WARPSIZE));

  // process set elements
  

  for (int n=get_local_id(0)+get_group_id(0)*get_local_size(0);
       n<set_size; n+=get_local_size(0)*get_num_groups(0)) {

    int offset = n - tid;
    int nelems = MIN(OP_WARPSIZE,set_size-offset);

    // copy data into shared memory, then into local

    for (int m=0; m<4; m++)
      arg_s[tid+m*nelems] = arg0[tid+m*nelems+offset*4];

    for (int m=0; m<4; m++)
      arg0_l[m] = arg_s[m+tid*4];


    // user-supplied kernel call

    save_soln( arg0_l,
               arg1_l );

    // copy back into shared memory, then to device

    for (int m=0; m<4; m++)
      arg_s[m+tid*4] = arg1_l[m];

    for (int m=0; m<4; m++)
      arg1[tid+m*nelems+offset*4] = arg_s[tid+m*nelems];
//     arg1[0]=9.999;

  }
}
