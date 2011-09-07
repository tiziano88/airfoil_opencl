struct global_constants {
  float gam;
  float gm1;
  float cfl;
  float eps;
  float mach;
  float alpha;
  float qinf[4];
};
#define ZERO_float 0.0f
#define ROUND_UP(bytes) (((bytes) + 15 ) & ~15 )
#define MIN(a,b) ((a<b) ? (a) : (b))
typedef enum {OP_READ, OP_WRITE, OP_RW, OP_INC, OP_MIN, OP_MAX} op_access;
inline void bres_calc(
  __local float *x1,  
  __local float *x2,  
  __local float *q1,
  __local float *adt1, 
  float *res1,
  __global int *bound,
  __constant struct global_constants *g_const_d ) {
  float dx,dy,mu, ri, p1,vol1, p2,vol2, f;

  dx = x1[0] - x2[0];
  dy = x1[1] - x2[1];

  ri = 1.0f/q1[0];
  p1 = g_const_d->gm1*(q1[3]-0.5f*ri*(q1[1]*q1[1]+q1[2]*q1[2]));

  if (*bound==1) {
    res1[1] += + p1*dy;
    res1[2] += - p1*dx;
  }
  else {
    vol1 =  ri*(q1[1]*dy - q1[2]*dx);

    ri   = 1.0f/g_const_d->qinf[0];
    p2   = g_const_d->gm1*(g_const_d->qinf[3]-0.5f*ri*(g_const_d->qinf[1]*g_const_d->qinf[1]+g_const_d->qinf[2]*g_const_d->qinf[2]));
    vol2 =  ri*(g_const_d->qinf[1]*dy - g_const_d->qinf[2]*dx);

    mu = (*adt1)*g_const_d->eps;

    f = 0.5f*(vol1* q1[0]         + vol2* g_const_d->qinf[0]        ) + mu*(q1[0]-g_const_d->qinf[0]);
    res1[0] += f;
    f = 0.5f*(vol1* q1[1] + p1*dy + vol2* g_const_d->qinf[1] + p2*dy) + mu*(q1[1]-g_const_d->qinf[1]);
    res1[1] += f;
    f = 0.5f*(vol1* q1[2] - p1*dx + vol2* g_const_d->qinf[2] - p2*dx) + mu*(q1[2]-g_const_d->qinf[2]);
    res1[2] += f;
    f = 0.5f*(vol1*(q1[3]+p1)     + vol2*(g_const_d->qinf[3]+p2)    ) + mu*(q1[3]-g_const_d->qinf[3]);
    res1[3] += f;
  }
}
__kernel void op_cuda_bres_calc(
  __global float *ind_arg0,
  __global int *ind_arg0_maps,
  __global float *ind_arg1,
  __global int *ind_arg1_maps,
  __global float *ind_arg2,
  __global int *ind_arg2_maps,
  __global float *ind_arg3,
  __global int *ind_arg3_maps,
  __global short *arg0_maps,
  __global short *arg1_maps,
  __global short *arg2_maps,
  __global short *arg3_maps,
  __global short *arg4_maps,
  __global int *arg5,
  __global int   *ind_arg_sizes,
  __global int   *ind_arg_offs,
  int    block_offset,
  __global int   *blkmap,
  __global int   *offset,
  __global int   *nelems,
  __global int   *ncolors,
  __global int   *colors,
  __local  float  *shared, 
  __constant struct global_constants *g_const_d ) {

  float arg4_l[4];

  __global int   * __local ind_arg0_map, * __local ind_arg1_map, * __local ind_arg2_map, * __local ind_arg3_map;
  __local int ind_arg0_size, ind_arg1_size, ind_arg2_size, ind_arg3_size;
  __local float * __local ind_arg0_s;
  __local float * __local ind_arg1_s;
  __local float * __local ind_arg2_s;
  __local float * __local ind_arg3_s;
  __local int    nelems2, ncolor;
  __local int    nelem, offset_b;

  if (get_local_id(0)==0) {

    // get sizes and shift pointers and direct-mapped data

    int blockId = blkmap[get_group_id(0) + block_offset];

    nelem    = nelems[blockId];
    offset_b = offset[blockId];

    nelems2  = get_local_size(0)*(1+(nelem-1)/get_local_size(0));
    ncolor   = ncolors[blockId];

    ind_arg0_size = ind_arg_sizes[0+blockId*4];
    ind_arg1_size = ind_arg_sizes[1+blockId*4];
    ind_arg2_size = ind_arg_sizes[2+blockId*4];
    ind_arg3_size = ind_arg_sizes[3+blockId*4];

    ind_arg0_map = ind_arg0_maps + ind_arg_offs[0+blockId*4];
    ind_arg1_map = ind_arg1_maps + ind_arg_offs[1+blockId*4];
    ind_arg2_map = ind_arg2_maps + ind_arg_offs[2+blockId*4];
    ind_arg3_map = ind_arg3_maps + ind_arg_offs[3+blockId*4];

    // set shared memory pointers
    int nelems = 0;
    ind_arg0_s = &shared[nelems];
    nelems    += ROUND_UP(ind_arg0_size*2);
    ind_arg1_s = &shared[nelems];
    nelems    += ROUND_UP(ind_arg1_size*4);
    ind_arg2_s = &shared[nelems];
    nelems    += ROUND_UP(ind_arg2_size*1);
    ind_arg3_s = &shared[nelems];
  }

  barrier( CLK_LOCAL_MEM_FENCE ); 

  // copy indirect datasets into shared memory or zero increment
  for (int n=get_local_id(0); n<ind_arg0_size*2; n+=get_local_size(0))
    ind_arg0_s[n] = ind_arg0[n%2+ind_arg0_map[n/2]*2];

  for (int n=get_local_id(0); n<ind_arg1_size*4; n+=get_local_size(0))
    ind_arg1_s[n] = ind_arg1[n%4+ind_arg1_map[n/4]*4];

  for (int n=get_local_id(0); n<ind_arg2_size*1; n+=get_local_size(0))
    ind_arg2_s[n] = ind_arg2[n%1+ind_arg2_map[n/1]*1];

  for (int n=get_local_id(0); n<ind_arg3_size*4; n+=get_local_size(0))
    ind_arg3_s[n] = ZERO_float;

  barrier( CLK_LOCAL_MEM_FENCE );

  // process set elements
  for (int n=get_local_id(0); n<nelems2; n+=get_local_size(0)) {
    int col2 = -1;

    if (n<nelem) {

      // initialise local variables
      for (int d=0; d<4; d++)
        arg4_l[d] = ZERO_float;

      // user-supplied kernel call
      bres_calc( ind_arg0_s+arg0_maps[n + offset_b]*2,
                 ind_arg0_s+arg1_maps[n + offset_b]*2,
                 ind_arg1_s+arg2_maps[n + offset_b]*4,
                 ind_arg2_s+arg3_maps[n + offset_b]*1,
                 arg4_l,
                 arg5+(n + offset_b)*1, g_const_d);

      col2 = colors[n + offset_b];
    }

    // store local variables
    int arg4_map = arg4_maps[n + offset_b];

    for (int col=0; col<ncolor; col++) {
      if (col2==col) {
        for (int d=0; d<4; d++)
          ind_arg3_s[d+arg4_map*4] += arg4_l[d];
      }
      barrier( CLK_LOCAL_MEM_FENCE );
    }
  }

  // apply pointered write/increment
  for (int n=get_local_id(0); n<ind_arg3_size*4; n+=get_local_size(0))
    ind_arg3[n%4+ind_arg3_map[n/4]*4] += ind_arg3_s[n];
}
//template < op_access reduction, class T >
inline void op_reduction( __global volatile float *dat_g, float dat_l, int reduction, __local float *temp)
{
  int tid = get_local_id( 0 );
  int d   = get_local_size( 0 )>>1; 

  barrier( CLK_LOCAL_MEM_FENCE );

  temp[tid] = dat_l;

  size_t warpSize = OP_WARPSIZE;

  for ( ; d>warpSize; d>>=1 ) {
    barrier( CLK_LOCAL_MEM_FENCE );
    if ( tid<d ) {
      switch ( reduction ) {
      case OP_INC:
        temp[tid] = temp[tid] + temp[tid+d];
        break;
      case OP_MIN:
        if( temp[tid+d]<temp[tid] ) temp[tid] = temp[tid+d];
        break;
      case OP_MAX:
        if( temp[tid+d]>temp[tid] ) temp[tid] = temp[tid+d];
        break;
      }
    }
  }

  barrier( CLK_LOCAL_MEM_FENCE );

  __local volatile float *vtemp = temp;   // see Fermi compatibility guide 

  if ( tid<warpSize ) {
    for ( ; d>0; d>>=1 ) {
      if ( tid<d ) {
        switch ( reduction ) {
        case OP_INC:
          vtemp[tid] = vtemp[tid] + vtemp[tid+d];
          break;
        case OP_MIN:
          if( vtemp[tid+d]<vtemp[tid] ) vtemp[tid] = vtemp[tid+d];
          break;
        case OP_MAX:
          if( vtemp[tid+d]>vtemp[tid] ) vtemp[tid] = vtemp[tid+d];
          break;
        }
      }
    }
  }

  if ( tid==0 ) {
    switch ( reduction ) {
    case OP_INC:
      *dat_g = *dat_g + vtemp[0];
      break;
    case OP_MIN:
      if( temp[0]<*dat_g ) *dat_g = vtemp[0];
      break;
    case OP_MAX:
      if( temp[0]>*dat_g ) *dat_g = vtemp[0];
      break;
    }
  }

}

inline void update(float *qold, float *q, float *res, __global float *adt, float *rms){
  float del, adti;

  adti = 1.0f/(*adt);

  for (int n=0; n<4; n++) {
    del    = adti*res[n];
    q[n]   = qold[n] - del;
    res[n] = 0.0f;
    *rms  += del*del;
  }
}

__kernel void op_cuda_update(
  __global float *arg0,
  __global float *arg1,
  __global float *arg2,
  __global float *arg3,
  __global float *arg4,
  int arg4_offset,
  int   offset_s,
  int   set_size,
  __local  float *shared ) {

  arg4 = arg4 + arg4_offset/sizeof(float);

  float arg0_l[4];
  float arg1_l[4];
  float arg2_l[4];
  float arg4_l[1];
  for (int d=0; d<1; d++) 
    arg4_l[d]=ZERO_float;


  int   tid = get_local_id(0)%OP_WARPSIZE;

  __local float *arg_s =  shared+ offset_s *(get_local_id(0)/OP_WARPSIZE)/sizeof(float);

  // process set elements
  for (int n=get_global_id(0); n<set_size; n+=get_global_size(0)) {

    int offset = n - tid;
    int nelems = MIN(OP_WARPSIZE,set_size-offset);

    // copy data into shared memory, then into local
    for (int m=0; m<4; m++)
      arg_s[tid+m*nelems] = arg0[tid+m*nelems+offset*4];

    for (int m=0; m<4; m++)
      arg0_l[m] = arg_s[m+tid*4];

    for (int m=0; m<4; m++)
      arg_s[tid+m*nelems] = arg2[tid+m*nelems+offset*4];

    for (int m=0; m<4; m++)
      arg2_l[m] = arg_s[m+tid*4];


    // user-supplied kernel call
    update( arg0_l,
            arg1_l,
            arg2_l,
            arg3+n,
            arg4_l );
    // copy back into shared memory, then to device
    for (int m=0; m<4; m++)
      arg_s[m+tid*4] = arg1_l[m];

    for (int m=0; m<4; m++)
      arg1[tid+m*nelems+offset*4] = arg_s[tid+m*nelems];

    for (int m=0; m<4; m++)
      arg_s[m+tid*4] = arg2_l[m];

    for (int m=0; m<4; m++)
      arg2[tid+m*nelems+offset*4] = arg_s[tid+m*nelems];


  }

  // global reductions
  __local float temp[1600];

  for(int d=0; d<1; d++)
    op_reduction(&arg4[d+get_group_id(0)*1],arg4_l[d],OP_INC, temp);
}


inline void adt_calc(__local float *x1,__local float *x2,__local float *x3,__local float *x4,__global float *q,__global float *adt, __constant struct global_constants *g_const_d){
  float dx,dy, ri,u,v,c;

  ri =  1.0f/q[0];
  u  =   ri*q[1];
  v  =   ri*q[2];
  c  = sqrt(g_const_d->gam*g_const_d->gm1*(ri*q[3]-0.5f*(u*u+v*v)));

  dx = x2[0] - x1[0];
  dy = x2[1] - x1[1];
  *adt  = fabs(u*dy-v*dx) + c*sqrt(dx*dx+dy*dy);

  dx = x3[0] - x2[0];
  dy = x3[1] - x2[1];
  *adt += fabs(u*dy-v*dx) + c*sqrt(dx*dx+dy*dy);

  dx = x4[0] - x3[0];
  dy = x4[1] - x3[1];
  *adt += fabs(u*dy-v*dx) + c*sqrt(dx*dx+dy*dy);

  dx = x1[0] - x4[0];
  dy = x1[1] - x4[1];
  *adt += fabs(u*dy-v*dx) + c*sqrt(dx*dx+dy*dy);

  *adt = (*adt) / g_const_d->cfl;
}

__kernel void op_cuda_adt_calc(
  __global float *ind_arg0,
  __global int *ind_arg0_maps,
  __global short *arg0_maps,
  __global short *arg1_maps,
  __global short *arg2_maps,
  __global short *arg3_maps,
  __global float *arg4,
  __global float *arg5,
  __global int   *ind_arg_sizes,
  __global int   *ind_arg_offs,
  int    block_offset,
  __global int   *blkmap,
  __global int   *offset,
  __global int   *nelems,
  __global int   *ncolors,
  __global int   *colors,
  __local  float  *shared,
  __constant struct global_constants *g_const_d ) {



  __global int   * __local ind_arg0_map;
  __local int ind_arg0_size;
  __local float * __local ind_arg0_s;
  __local int    nelem, offset_b;

  if ( get_local_id( 0 ) == 0 ) {

    // get sizes and shift pointers and direct-mapped data
    int blockId = blkmap[ get_group_id( 0 ) + block_offset ];

    nelem    = nelems[blockId];
    offset_b = offset[blockId];

    ind_arg0_size = ind_arg_sizes[ 0 + blockId * 1 ];

    ind_arg0_map = ind_arg0_maps + ind_arg_offs[ 0 + blockId * 1 ];

    // set shared memory pointers
    int nbytes = 0;
    ind_arg0_s = shared + nbytes/4;
  }

  barrier( CLK_LOCAL_MEM_FENCE );

  // copy indirect datasets into shared memory or zero increment
  for (int n=get_local_id(0); n<ind_arg0_size*2; n+=get_local_size(0))
    ind_arg0_s[n] = ind_arg0[n%2+ind_arg0_map[n/2]*2];

  barrier( CLK_LOCAL_MEM_FENCE );

  // process set elements
  for (int n=get_local_id(0); n<nelem; n+=get_local_size(0)) {

      // user-supplied kernel call
      adt_calc( ind_arg0_s+arg0_maps[n+offset_b]*2,
                ind_arg0_s+arg1_maps[n+offset_b]*2,
                ind_arg0_s+arg2_maps[n+offset_b]*2,
                ind_arg0_s+arg3_maps[n+offset_b]*2,
                arg4+(n+offset_b)*4,
                arg5+(n+offset_b)*1,
                g_const_d );
  }

}

inline void save_soln(float *q, float *qold){
 for (int n=0; n<4; n++) qold[n] = q[n];
}

__kernel void op_cuda_save_soln(
  __global float *arg0,
  __global float *arg1,
  int   offset_s,
  int   set_size ,
  __local  char *shared ) {

  float arg0_l[4];
  float arg1_l[4];
  int   tid = get_local_id(0) % OP_WARPSIZE;

  __local float *arg_s = (__local float *) (shared+ offset_s *(get_local_id(0)/OP_WARPSIZE));


  // process set elements
  

  //for (int n=get_local_id(0)+get_group_id(0)*get_local_size(0);
  //     n<set_size; n+=get_local_size(0)*get_num_groups(0)) {
  for (int n=get_global_id(0); n<set_size; n+=get_global_size(0)) {

    int offset = n - tid;
    int nelems = MIN(OP_WARPSIZE,set_size-offset);

    // copy data into shared memory, then into local

    
    for (int m=0; m<4; m++)
      arg_s[tid+m*nelems] = arg0[offset*4 + tid+m*nelems];


    for (int m=0; m<4; m++)
      arg0_l[m] = arg_s[tid*4 + m];
      
/*
    for (int m=0; m<4; ++m) {
      arg0_l[m] = arg0[n*4+m];
    }
    */


    // user-supplied kernel call

    save_soln( arg0_l,
               arg1_l );

/*
    for (int m=0; m<4; ++m) {
      arg1[n*4+m] = arg1_l[m];
    }
    */

    // copy back into shared memory, then to device

    
    for (int m=0; m<4; m++)
      arg_s[m+tid*4] = arg1_l[m];

    for (int m=0; m<4; m++)
      arg1[tid+m*nelems+offset*4] = arg_s[tid+m*nelems];
      
      

  }
}


inline void res_calc(
  VECTYPE *x1, 
  VECTYPE *x2, 
  VECTYPE *q1, 
  VECTYPE *q2, 
  VECTYPE *adt1, 
  VECTYPE *adt2, 
  VECTYPE *res1, 
  VECTYPE *res2, 
  __constant struct global_constants *g_const_d) {
  VECTYPE dx,dy,mu, ri, p1,vol1, p2,vol2, f;
  dx = x1[0] - x2[0];
  dy = x1[1] - x2[1];
  ri   = 1.0f/q1[0];
  p1   = g_const_d->gm1*(q1[3]-0.5f*ri*(q1[1]*q1[1]+q1[2]*q1[2]));
  vol1 =  ri*(q1[1]*dy - q1[2]*dx);

  ri   = 1.0f/q2[0];
  p2   = g_const_d->gm1*(q2[3]-0.5f*ri*(q2[1]*q2[1]+q2[2]*q2[2]));
  vol2 =  ri*(q2[1]*dy - q2[2]*dx);

  mu = 0.5f*((*adt1)+(*adt2))*g_const_d->eps;

  f = 0.5f*(vol1* q1[0]         + vol2* q2[0]        ) + mu*(q1[0]-q2[0]);
  res1[0] += f;
  res2[0] -= f;
  f = 0.5f*(vol1* q1[1] + p1*dy + vol2* q2[1] + p2*dy) + mu*(q1[1]-q2[1]);
  res1[1] += f;
  res2[1] -= f;
  f = 0.5f*(vol1* q1[2] - p1*dx + vol2* q2[2] - p2*dx) + mu*(q1[2]-q2[2]);
  res1[2] += f;
  res2[2] -= f;
  f = 0.5f*(vol1*(q1[3]+p1)     + vol2*(q2[3]+p2)    ) + mu*(q1[3]-q2[3]);
  res1[3] += f;
  res2[3] -= f;
}

__kernel void op_cuda_res_calc(
  __global float *ind_arg0,
  __global int *ind_arg0_maps,
  __global float *ind_arg1,
  __global int *ind_arg1_maps,
  __global float *ind_arg2,
  __global int *ind_arg2_maps,
  __global float *ind_arg3,
  __global int *ind_arg3_maps,
  __global short *arg0_maps,
  __global short *arg1_maps,
  __global short *arg2_maps,
  __global short *arg3_maps,
  __global short *arg4_maps,
  __global short *arg5_maps,
  __global short *arg6_maps,
  __global short *arg7_maps,
  __global int   *ind_arg_sizes,
  __global int   *ind_arg_offs,
  int    block_offset,
  __global int   *blkmap,
  __global int   *offset,
  __global int   *nelems,
  __global int   *ncolors,
  __global int   *colors,
  __local  float *shared, 
  __constant struct global_constants *g_const_d) {

  VECTYPE arg0_l[2];
  VECTYPE arg1_l[2];
  VECTYPE arg2_l[4];
  VECTYPE arg3_l[4];
  VECTYPE arg4_l[1];
  VECTYPE arg5_l[1];
  VECTYPE arg6_l[4];
  VECTYPE arg7_l[4];

  __global int * __local ind_arg0_map, * __local ind_arg1_map, * __local ind_arg2_map, * __local ind_arg3_map;
  __local int ind_arg0_size, ind_arg1_size, ind_arg2_size, ind_arg3_size;
  __local float * __local ind_arg0_s;
  __local float * __local ind_arg1_s;
  __local float * __local ind_arg2_s;
  __local float * __local ind_arg3_s;
  __local int    nelems2, ncolor;
  __local int    nelem, offset_b;

  if (get_local_id(0)==0) {

    // get sizes and shift pointers and direct-mapped data

    int blockId = blkmap[get_group_id(0) + block_offset];

    nelem    = nelems[blockId];
    offset_b = offset[blockId];

    nelems2  = get_local_size(0)*(1+(nelem-1)/get_local_size(0));
    ncolor   = ncolors[blockId];

    
    ind_arg0_size = ind_arg_sizes[0+blockId*4];
    ind_arg1_size = ind_arg_sizes[1+blockId*4];
    ind_arg2_size = ind_arg_sizes[2+blockId*4];
    ind_arg3_size = ind_arg_sizes[3+blockId*4];

    ind_arg0_map = ind_arg0_maps + ind_arg_offs[0+blockId*4];
    ind_arg1_map = ind_arg1_maps + ind_arg_offs[1+blockId*4];
    ind_arg2_map = ind_arg2_maps + ind_arg_offs[2+blockId*4];
    ind_arg3_map = ind_arg3_maps + ind_arg_offs[3+blockId*4];

    // set shared memory pointers

    int nbytes = 0;
    ind_arg0_s = shared + nbytes/sizeof(float);
    nbytes    += ROUND_UP(ind_arg0_size*sizeof(float)*2);
    ind_arg1_s = shared + nbytes/sizeof(float);
    nbytes    += ROUND_UP(ind_arg1_size*sizeof(float)*4);
    ind_arg2_s = shared + nbytes/sizeof(float);
    nbytes    += ROUND_UP(ind_arg2_size*sizeof(float)*1);
    ind_arg3_s = shared + nbytes/sizeof(float);
  }
  barrier( CLK_LOCAL_MEM_FENCE ); // make sure all of above completed
  // copy indirect datasets into shared memory or zero increment
  for (int n=get_local_id(0); n<ind_arg0_size*2; n+=get_local_size(0))
    ind_arg0_s[n] = ind_arg0[n%2+ind_arg0_map[n/2]*2];
  for (int n=get_local_id(0); n<ind_arg1_size*4; n+=get_local_size(0))
    ind_arg1_s[n] = ind_arg1[n%4+ind_arg1_map[n/4]*4];
  for (int n=get_local_id(0); n<ind_arg2_size*1; n+=get_local_size(0))
    ind_arg2_s[n] = ind_arg2[n%1+ind_arg2_map[n/1]*1];
  for (int n=get_local_id(0); n<ind_arg3_size*4; n+=get_local_size(0))
    ind_arg3_s[n] = 0.0f;
  barrier( CLK_LOCAL_MEM_FENCE );

  if (nelem != 128 ) {
  //printf("%d/%d/%d\n", nelems2, nelem, get_local_size(0));
  }


  // process set elements
  for (int n=get_local_id(0)*VEC; n<nelems2; n+=get_local_size(0)*VEC) {
    int col2 = -1;
    if (n<nelem) {

      for (int m=0; m<2; m++) { 
#if VEC>1
        arg0_l[m].s0 = ind_arg0_s[arg0_maps[n+0x0 + offset_b]*2+m];
        arg0_l[m].s1 = ind_arg0_s[arg0_maps[n+0x1 + offset_b]*2+m];
#if VEC>2
        arg0_l[m].s2 = ind_arg0_s[arg0_maps[n+0x2 + offset_b]*2+m];
        arg0_l[m].s3 = ind_arg0_s[arg0_maps[n+0x3 + offset_b]*2+m];
#if VEC>4
        arg0_l[m].s4 = ind_arg0_s[arg0_maps[n+0x4 + offset_b]*2+m];
        arg0_l[m].s5 = ind_arg0_s[arg0_maps[n+0x5 + offset_b]*2+m];
        arg0_l[m].s6 = ind_arg0_s[arg0_maps[n+0x6 + offset_b]*2+m];
        arg0_l[m].s7 = ind_arg0_s[arg0_maps[n+0x7 + offset_b]*2+m];
#if VEC>8
        arg0_l[m].s8 = ind_arg0_s[arg0_maps[n+0x8 + offset_b]*2+m];
        arg0_l[m].s9 = ind_arg0_s[arg0_maps[n+0x9 + offset_b]*2+m];
        arg0_l[m].sa = ind_arg0_s[arg0_maps[n+0xa + offset_b]*2+m];
        arg0_l[m].sb = ind_arg0_s[arg0_maps[n+0xb + offset_b]*2+m];
        arg0_l[m].sc = ind_arg0_s[arg0_maps[n+0xc + offset_b]*2+m];
        arg0_l[m].sd = ind_arg0_s[arg0_maps[n+0xd + offset_b]*2+m];
        arg0_l[m].se = ind_arg0_s[arg0_maps[n+0xd + offset_b]*2+m];
        arg0_l[m].sf = ind_arg0_s[arg0_maps[n+0xf + offset_b]*2+m];
#endif
#endif
#endif
#else
        arg0_l[m].s0 = ind_arg0_s[arg0_maps[n + offset_b]*2+m];
#endif
      }

      for (int m=0; m<2; m++) { 
#if VEC>1
        arg1_l[m].s0 = ind_arg0_s[arg1_maps[n+0x0 + offset_b]*2+m];
        arg1_l[m].s1 = ind_arg0_s[arg1_maps[n+0x1 + offset_b]*2+m];
#if VEC>2
        arg1_l[m].s2 = ind_arg0_s[arg1_maps[n+0x2 + offset_b]*2+m];
        arg1_l[m].s3 = ind_arg0_s[arg1_maps[n+0x3 + offset_b]*2+m];
#if VEC>4
        arg1_l[m].s4 = ind_arg0_s[arg1_maps[n+0x4 + offset_b]*2+m];
        arg1_l[m].s5 = ind_arg0_s[arg1_maps[n+0x5 + offset_b]*2+m];
        arg1_l[m].s6 = ind_arg0_s[arg1_maps[n+0x6 + offset_b]*2+m];
        arg1_l[m].s7 = ind_arg0_s[arg1_maps[n+0x7 + offset_b]*2+m];
#if VEC>8
        arg1_l[m].s8 = ind_arg0_s[arg1_maps[n+0x8 + offset_b]*2+m];
        arg1_l[m].s9 = ind_arg0_s[arg1_maps[n+0x9 + offset_b]*2+m];
        arg1_l[m].sa = ind_arg0_s[arg1_maps[n+0xa + offset_b]*2+m];
        arg1_l[m].sb = ind_arg0_s[arg1_maps[n+0xb + offset_b]*2+m];
        arg1_l[m].sc = ind_arg0_s[arg1_maps[n+0xc + offset_b]*2+m];
        arg1_l[m].sd = ind_arg0_s[arg1_maps[n+0xd + offset_b]*2+m];
        arg1_l[m].se = ind_arg0_s[arg1_maps[n+0xe + offset_b]*2+m];
        arg1_l[m].sf = ind_arg0_s[arg1_maps[n+0xf + offset_b]*2+m];
#endif
#endif
#endif
#else
        arg1_l[m].s0 = ind_arg0_s[arg1_maps[n + offset_b]*2+m];
#endif
      }

      for (int m=0; m<4; m++) { 
#if VEC>1
        arg2_l[m].s0 = ind_arg1_s[arg2_maps[n+0x0 + offset_b]*4+m];
        arg2_l[m].s1 = ind_arg1_s[arg2_maps[n+0x1 + offset_b]*4+m];
#if VEC>2
        arg2_l[m].s2 = ind_arg1_s[arg2_maps[n+0x2 + offset_b]*4+m];
        arg2_l[m].s3 = ind_arg1_s[arg2_maps[n+0x3 + offset_b]*4+m];
#if VEC>4
        arg2_l[m].s4 = ind_arg1_s[arg2_maps[n+0x4 + offset_b]*4+m];
        arg2_l[m].s5 = ind_arg1_s[arg2_maps[n+0x5 + offset_b]*4+m];
        arg2_l[m].s6 = ind_arg1_s[arg2_maps[n+0x6 + offset_b]*4+m];
        arg2_l[m].s7 = ind_arg1_s[arg2_maps[n+0x7 + offset_b]*4+m];
#if VEC>8
        arg2_l[m].s8 = ind_arg1_s[arg2_maps[n+0x8 + offset_b]*4+m];
        arg2_l[m].s9 = ind_arg1_s[arg2_maps[n+0x9 + offset_b]*4+m];
        arg2_l[m].sa = ind_arg1_s[arg2_maps[n+0xa + offset_b]*4+m];
        arg2_l[m].sb = ind_arg1_s[arg2_maps[n+0xb + offset_b]*4+m];
        arg2_l[m].sc = ind_arg1_s[arg2_maps[n+0xc + offset_b]*4+m];
        arg2_l[m].sd = ind_arg1_s[arg2_maps[n+0xd + offset_b]*4+m];
        arg2_l[m].se = ind_arg1_s[arg2_maps[n+0xe + offset_b]*4+m];
        arg2_l[m].sf = ind_arg1_s[arg2_maps[n+0xf + offset_b]*4+m];
#endif
#endif
#endif
#else
        arg2_l[m].s0 = ind_arg1_s[arg2_maps[n + offset_b]*4+m];
#endif
      }

      for (int m=0; m<4; m++) { 
#if VEC>1
        arg3_l[m].s0 = ind_arg1_s[arg3_maps[n+0x0 + offset_b]*4+m];
        arg3_l[m].s1 = ind_arg1_s[arg3_maps[n+0x1 + offset_b]*4+m];
#if VEC>2
        arg3_l[m].s2 = ind_arg1_s[arg3_maps[n+0x2 + offset_b]*4+m];
        arg3_l[m].s3 = ind_arg1_s[arg3_maps[n+0x3 + offset_b]*4+m];
#if VEC>4
        arg3_l[m].s4 = ind_arg1_s[arg3_maps[n+0x4 + offset_b]*4+m];
        arg3_l[m].s5 = ind_arg1_s[arg3_maps[n+0x5 + offset_b]*4+m];
        arg3_l[m].s6 = ind_arg1_s[arg3_maps[n+0x6 + offset_b]*4+m];
        arg3_l[m].s7 = ind_arg1_s[arg3_maps[n+0x7 + offset_b]*4+m];
#if VEC>8
        arg3_l[m].s8 = ind_arg1_s[arg3_maps[n+0x8 + offset_b]*4+m];
        arg3_l[m].s9 = ind_arg1_s[arg3_maps[n+0x9 + offset_b]*4+m];
        arg3_l[m].sa = ind_arg1_s[arg3_maps[n+0xa + offset_b]*4+m];
        arg3_l[m].sb = ind_arg1_s[arg3_maps[n+0xb + offset_b]*4+m];
        arg3_l[m].sc = ind_arg1_s[arg3_maps[n+0xc + offset_b]*4+m];
        arg3_l[m].sd = ind_arg1_s[arg3_maps[n+0xd + offset_b]*4+m];
        arg3_l[m].se = ind_arg1_s[arg3_maps[n+0xe + offset_b]*4+m];
        arg3_l[m].sf = ind_arg1_s[arg3_maps[n+0xf + offset_b]*4+m];
#endif
#endif
#endif
#else
        arg3_l[m].s0 = ind_arg1_s[arg3_maps[n + offset_b]*4+m];
#endif
      }

      for (int m=0; m<1; m++) { 
#if VEC>1
        arg4_l[m].s0 = ind_arg2_s[arg4_maps[n+0x0 + offset_b]*1+m];
        arg4_l[m].s1 = ind_arg2_s[arg4_maps[n+0x1 + offset_b]*1+m];
#if VEC>2
        arg4_l[m].s2 = ind_arg2_s[arg4_maps[n+0x2 + offset_b]*1+m];
        arg4_l[m].s3 = ind_arg2_s[arg4_maps[n+0x3 + offset_b]*1+m];
#if VEC>4
        arg4_l[m].s4 = ind_arg2_s[arg4_maps[n+0x4 + offset_b]*1+m];
        arg4_l[m].s5 = ind_arg2_s[arg4_maps[n+0x5 + offset_b]*1+m];
        arg4_l[m].s6 = ind_arg2_s[arg4_maps[n+0x6 + offset_b]*1+m];
        arg4_l[m].s7 = ind_arg2_s[arg4_maps[n+0x7 + offset_b]*1+m];
#if VEC>8
        arg4_l[m].s8 = ind_arg2_s[arg4_maps[n+0x8 + offset_b]*1+m];
        arg4_l[m].s9 = ind_arg2_s[arg4_maps[n+0x9 + offset_b]*1+m];
        arg4_l[m].sa = ind_arg2_s[arg4_maps[n+0xa + offset_b]*1+m];
        arg4_l[m].sb = ind_arg2_s[arg4_maps[n+0xb + offset_b]*1+m];
        arg4_l[m].sc = ind_arg2_s[arg4_maps[n+0xc + offset_b]*1+m];
        arg4_l[m].sd = ind_arg2_s[arg4_maps[n+0xd + offset_b]*1+m];
        arg4_l[m].se = ind_arg2_s[arg4_maps[n+0xe + offset_b]*1+m];
        arg4_l[m].sf = ind_arg2_s[arg4_maps[n+0xf + offset_b]*1+m];
#endif
#endif
#endif
#else
        arg4_l[m].s0 = ind_arg2_s[arg4_maps[n + offset_b]*1+m];
#endif
      }

      for (int m=0; m<1; m++) { 
#if VEC>1
        arg5_l[m].s0 = ind_arg2_s[arg5_maps[n+0x0 + offset_b]*1+m];
        arg5_l[m].s1 = ind_arg2_s[arg5_maps[n+0x1 + offset_b]*1+m];
#if VEC>2
        arg5_l[m].s2 = ind_arg2_s[arg5_maps[n+0x2 + offset_b]*1+m];
        arg5_l[m].s3 = ind_arg2_s[arg5_maps[n+0x3 + offset_b]*1+m];
#if VEC>4
        arg5_l[m].s4 = ind_arg2_s[arg5_maps[n+0x4 + offset_b]*1+m];
        arg5_l[m].s5 = ind_arg2_s[arg5_maps[n+0x5 + offset_b]*1+m];
        arg5_l[m].s6 = ind_arg2_s[arg5_maps[n+0x6 + offset_b]*1+m];
        arg5_l[m].s7 = ind_arg2_s[arg5_maps[n+0x7 + offset_b]*1+m];
#if VEC>8
        arg5_l[m].s8 = ind_arg2_s[arg5_maps[n+0x8 + offset_b]*1+m];
        arg5_l[m].s9 = ind_arg2_s[arg5_maps[n+0x9 + offset_b]*1+m];
        arg5_l[m].sa = ind_arg2_s[arg5_maps[n+0xa + offset_b]*1+m];
        arg5_l[m].sb = ind_arg2_s[arg5_maps[n+0xb + offset_b]*1+m];
        arg5_l[m].sc = ind_arg2_s[arg5_maps[n+0xc + offset_b]*1+m];
        arg5_l[m].sd = ind_arg2_s[arg5_maps[n+0xd + offset_b]*1+m];
        arg5_l[m].se = ind_arg2_s[arg5_maps[n+0xe + offset_b]*1+m];
        arg5_l[m].sf = ind_arg2_s[arg5_maps[n+0xf + offset_b]*1+m];
#endif
#endif
#endif
#else
        arg5_l[m].s0 = ind_arg2_s[arg5_maps[n + offset_b]*1+m];
#endif
      }

      // initialise local variables
      for (int m=0; m<4; m++) {
        arg6_l[m] = 0.0f;
      }
      for (int m=0; m<4; m++) {
        arg7_l[m] = 0.0f;
      }
      // user-supplied kernel call
      res_calc( arg0_l,
                arg1_l,
                arg2_l,
                arg3_l,
                arg4_l,
                arg5_l,
                arg6_l,
                arg7_l, 
                g_const_d );
      col2 = colors[n + offset_b];
    }

    // store local variables
    //int arg6_map = arg6_maps[n + offset_b];
    //int arg7_map = arg7_maps[n + offset_b];
    for (int col=0; col<ncolor; col++) {
      if (col2==col) {
        for (int d=0; d<4; d++) {
#if VEC>1
          ind_arg3_s[arg6_maps[n+0x0 + offset_b]*4+d] += arg6_l[d].s0;
          //ind_arg3_s[arg6_maps[n+0x1 + offset_b]*4+d] += arg6_l[d].s1;
#if VEC>2
          ind_arg3_s[arg6_maps[n+0x2 + offset_b]*4+d] += arg6_l[d].s2;
          ind_arg3_s[arg6_maps[n+0x3 + offset_b]*4+d] += arg6_l[d].s3;
#if VEC>4
          ind_arg3_s[arg6_maps[n+0x4 + offset_b]*4+d] += arg6_l[d].s4;
          ind_arg3_s[arg6_maps[n+0x5 + offset_b]*4+d] += arg6_l[d].s5;
          ind_arg3_s[arg6_maps[n+0x6 + offset_b]*4+d] += arg6_l[d].s6;
          ind_arg3_s[arg6_maps[n+0x7 + offset_b]*4+d] += arg6_l[d].s7;
#if VEC>8
          ind_arg3_s[arg6_maps[n+0x8 + offset_b]*4+d] += arg6_l[d].s8;
          ind_arg3_s[arg6_maps[n+0x9 + offset_b]*4+d] += arg6_l[d].s9;
          ind_arg3_s[arg6_maps[n+0xa + offset_b]*4+d] += arg6_l[d].sa;
          ind_arg3_s[arg6_maps[n+0xb + offset_b]*4+d] += arg6_l[d].sb;
          ind_arg3_s[arg6_maps[n+0xc + offset_b]*4+d] += arg6_l[d].sc;
          ind_arg3_s[arg6_maps[n+0xd + offset_b]*4+d] += arg6_l[d].sd;
          ind_arg3_s[arg6_maps[n+0xd + offset_b]*4+d] += arg6_l[d].sd;
          ind_arg3_s[arg6_maps[n+0xf + offset_b]*4+d] += arg6_l[d].sf;
#endif
#endif
#endif
#else
          ind_arg3_s[arg6_maps[ n + offset_b ] *4+d] += arg6_l[d].s0;
#endif
        }
        for (int d=0; d<4; d++) {
#if VEC>1
          ind_arg3_s[arg7_maps[n+0x0 + offset_b]*4+d] += arg7_l[d].s0;
          //ind_arg3_s[arg7_maps[n+0x1 + offset_b]*4+d] += arg7_l[d].s1;
#if VEC>2
          ind_arg3_s[arg7_maps[n+0x2 + offset_b]*4+d] += arg7_l[d].s2;
          ind_arg3_s[arg7_maps[n+0x3 + offset_b]*4+d] += arg7_l[d].s3;
#if VEC>4
          ind_arg3_s[arg7_maps[n+0x4 + offset_b]*4+d] += arg7_l[d].s4;
          ind_arg3_s[arg7_maps[n+0x5 + offset_b]*4+d] += arg7_l[d].s5;
          ind_arg3_s[arg7_maps[n+0x6 + offset_b]*4+d] += arg7_l[d].s6;
          ind_arg3_s[arg7_maps[n+0x7 + offset_b]*4+d] += arg7_l[d].s7;
#if VEC>8
          ind_arg3_s[arg7_maps[n+0x8 + offset_b]*4+d] += arg7_l[d].s8;
          ind_arg3_s[arg7_maps[n+0x9 + offset_b]*4+d] += arg7_l[d].s9;
          ind_arg3_s[arg7_maps[n+0xa + offset_b]*4+d] += arg7_l[d].sa;
          ind_arg3_s[arg7_maps[n+0xb + offset_b]*4+d] += arg7_l[d].sb;
          ind_arg3_s[arg7_maps[n+0xc + offset_b]*4+d] += arg7_l[d].sc;
          ind_arg3_s[arg7_maps[n+0xd + offset_b]*4+d] += arg7_l[d].sd;
          ind_arg3_s[arg7_maps[n+0xd + offset_b]*4+d] += arg7_l[d].sd;
          ind_arg3_s[arg7_maps[n+0xf + offset_b]*4+d] += arg7_l[d].sf;
#endif
#endif
#endif
#else
          ind_arg3_s[arg7_maps[ n + offset_b ] *4+d] += arg7_l[d].s0;
#endif
        }
      }
      barrier( CLK_LOCAL_MEM_FENCE );
    } 

  }
  // apply pointered write/increment
  for (int n=get_local_id(0); n<ind_arg3_size*4; n+=get_local_size(0))
    ind_arg3[n%4+ind_arg3_map[n/4]*4] += ind_arg3_s[n];
}
