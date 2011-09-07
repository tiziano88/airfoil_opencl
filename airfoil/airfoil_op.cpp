// 
// auto-generated by op2.m on 30-May-2011 22:03:11 
//

/*
  Open source copyright declaration based on BSD open source template:
  http://www.opensource.org/licenses/bsd-license.php

* Copyright (c) 2009-2011, Mike Giles
* All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
*     * Redistributions of source code must retain the above copyright
*       notice, this list of conditions and the following disclaimer.
*     * Redistributions in binary form must reproduce the above copyright
*       notice, this list of conditions and the following disclaimer in the
*       documentation and/or other materials provided with the distribution.
*     * The name of Mike Giles may not be used to endorse or promote products
*       derived from this software without specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY Mike Giles ''AS IS'' AND ANY
* EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
* WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
* DISCLAIMED. IN NO EVENT SHALL Mike Giles BE LIABLE FOR ANY
* DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
* (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
* LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
* ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
* (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
* SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

//
//     Nonlinear airfoil lift calculation
//
//     Written by Mike Giles, 2010-2011, based on FORTRAN code
//     by Devendra Ghate and Mike Giles, 2005
//

//
// standard headers
//

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#include <utility>
#define __NO_STD_VECTOR //use cl::vector
#include <CL/cl.h>
// global constants

//float gam, gm1, cfl, eps, mach, alpha, qinf[4];

struct global_constants {
  float gam;
  float gm1;
  float cfl;
  float eps;
  float mach;
  float alpha;
  float qinf[4];
};

struct global_constants g_const;
cl_mem g_const_d;

#include "op_lib.h"

//#define DIAGNOSTIC 1

//
// OP header file
//


//
// op_par_loop declarations
//

void op_par_loop_save_soln(char const *, op_set,
  op_arg,
  op_arg );

void op_par_loop_adt_calc(char const *, op_set,
  op_arg,
  op_arg,
  op_arg,
  op_arg,
  op_arg,
  op_arg );

void op_par_loop_res_calc(char const *, op_set,
  op_arg,
  op_arg,
  op_arg,
  op_arg,
  op_arg,
  op_arg,
  op_arg,
  op_arg );

void op_par_loop_bres_calc(char const *, op_set,
  op_arg,
  op_arg,
  op_arg,
  op_arg,
  op_arg,
  op_arg );

void op_par_loop_update(char const *, op_set,
  op_arg,
  op_arg,
  op_arg,
  op_arg,
  op_arg );

// kernel routines for parallel loops
//

/*
#include "save_soln.h"
#include "adt_calc.h"
#include "res_calc.h"
#include "bres_calc.h"
#include "update.h"
*/

// main program

void print_array( float *array, int len, const char *file ) {
  FILE *flog;
  flog = fopen( file, "w" );
  for( int i=0; i<len; ++i ) {
    fprintf( flog, "%f\n", array[i] );
  }
  fclose( flog );
}

void dump_array( op_dat dat, const char *file ) {
  op_fetch_data( dat );
  print_array( ( float *) dat->data, dat->set->size, file );
}

int main(int argc, char **argv){

  int    *becell, *ecell,  *bound, *bedge, *edge, *cell;
  float  *x, *q, *qold, *adt, *res;

  int    nnode,ncell,nedge,nbedge,niter;
  float  rms;

  // read in grid

  printf("reading in grid \n");

  FILE *fp;
  if ( (fp = fopen("new_grid.dat","r")) == NULL) {
    printf("can't open file new_grid.dat\n"); exit(-1);
  }

  if (fscanf(fp,"%d %d %d %d \n",&nnode, &ncell, &nedge, &nbedge) != 4) {
    printf("error reading from new_grid.dat\n"); exit(-1);
  }

  cell   = (int *) malloc(4*ncell*sizeof(int));
  edge   = (int *) malloc(2*nedge*sizeof(int));
  ecell  = (int *) malloc(2*nedge*sizeof(int));
  bedge  = (int *) malloc(2*nbedge*sizeof(int));
  becell = (int *) malloc(  nbedge*sizeof(int));
  bound  = (int *) malloc(  nbedge*sizeof(int));

  x      = (float *) malloc(2*nnode*sizeof(float));
  q      = (float *) malloc(4*ncell*sizeof(float));
  qold   = (float *) malloc(4*ncell*sizeof(float));
  res    = (float *) malloc(4*ncell*sizeof(float));
  adt    = (float *) malloc(  ncell*sizeof(float));

  for (int n=0; n<nnode; n++) {
    if (fscanf(fp,"%f %f \n",&x[2*n], &x[2*n+1]) != 2) {
      printf("error reading from new_grid.dat\n"); exit(-1);
    }
  }

  for (int n=0; n<ncell; n++) {
    if (fscanf(fp,"%d %d %d %d \n",&cell[4*n  ], &cell[4*n+1],
                                   &cell[4*n+2], &cell[4*n+3]) != 4) {
      printf("error reading from new_grid.dat\n"); exit(-1);
    }
  }

  for (int n=0; n<nedge; n++) {
    if (fscanf(fp,"%d %d %d %d \n",&edge[2*n], &edge[2*n+1],
                                   &ecell[2*n],&ecell[2*n+1]) != 4) {
      printf("error reading from new_grid.dat\n"); exit(-1);
    }
  }

  for (int n=0; n<nbedge; n++) {
    if (fscanf(fp,"%d %d %d %d \n",&bedge[2*n],&bedge[2*n+1],
                                   &becell[n], &bound[n]) != 4) {
      printf("error reading from new_grid.dat\n"); exit(-1);
    }
  }

  fclose(fp);

  // set constants and initialise flow field and residual

  printf("initialising flow field \n");

  g_const.gam = 1.4f;
  g_const.gm1 = g_const.gam - 1.0f;
  g_const.cfl = 0.9f;
  g_const.eps = 0.05f;

  g_const.mach  = 0.4f;
  g_const.alpha = 3.0f*atan(1.0f)/45.0f;  
  float p     = 1.0f;
  float r     = 1.0f;
  float u     = sqrt(g_const.gam*p/r)*g_const.mach;
  float e     = p/(r*g_const.gm1) + 0.5f*u*u;

  g_const.qinf[0] = r;
  g_const.qinf[1] = r*u;
  g_const.qinf[2] = 0.0f;
  g_const.qinf[3] = r*e;

  for (int n=0; n<ncell; n++) {
    for (int m=0; m<4; m++) {
        q[4*n+m] = g_const.qinf[m];
      res[4*n+m] = 0.0f;
    }
  }




  // OP initialisation

  printf("OP initialisation\n");
  op_init(argc,argv,2);
  g_const_d = op_allocate_constant( &g_const, sizeof( struct global_constants ) );

  // declare sets, pointers, datasets and global constants

  op_set nodes  = op_decl_set(nnode,  "nodes");
  op_set edges  = op_decl_set(nedge,  "edges");
  op_set bedges = op_decl_set(nbedge, "bedges");
  op_set cells  = op_decl_set(ncell,  "cells");

  op_map pedge   = op_decl_map(edges, nodes,2,edge,  "pedge");
  op_map pecell  = op_decl_map(edges, cells,2,ecell, "pecell");
  op_map pbedge  = op_decl_map(bedges,nodes,2,bedge, "pbedge");
  op_map pbecell = op_decl_map(bedges,cells,1,becell,"pbecell");
  op_map pcell   = op_decl_map(cells, nodes,4,cell,  "pcell");

  op_dat p_bound = op_decl_dat(bedges,1,"int"  ,bound,"p_bound");
  op_dat p_x     = op_decl_dat(nodes ,2,"float",x    ,"p_x");
  op_dat p_q     = op_decl_dat(cells ,4,"float",q    ,"p_q");
  op_dat p_qold  = op_decl_dat(cells ,4,"float",qold ,"p_qold");
  op_dat p_adt   = op_decl_dat(cells ,1,"float",adt  ,"p_adt");
  op_dat p_res   = op_decl_dat(cells ,4,"float",res  ,"p_res");

  op_decl_const2("gam",1,"float",&g_const.gam  );
  op_decl_const2("gm1",1,"float",&g_const.gm1  );
  op_decl_const2("cfl",1,"float",&g_const.cfl  );
  op_decl_const2("eps",1,"float",&g_const.eps  );
  op_decl_const2("mach",1,"float",&g_const.mach );
  op_decl_const2("alpha",1,"float",&g_const.alpha);
  op_decl_const2("qinf",4,"float",g_const.qinf  );


  op_diagnostic_output();

// main time-marching loop

  niter = 1000;

  for(int iter=1; iter<=niter; iter++) {

//  save old flow solution



    op_par_loop_save_soln("save_soln", cells,
                op_arg_dat(p_q,   -1,OP_ID, 4,"float",OP_READ ),
                op_arg_dat(p_qold,-1,OP_ID, 4,"float",OP_WRITE));

/*    if ( iter == 1 ) {
      dump_array( p_qold, "p_qold" );
    }
    */

#ifdef DIAGNOSTIC
    if (iter==1) {
      dump_array( p_qold, "p_qold" );
    }
#endif
    //dump_array( p_qold, "p_qold" );
    //op_fetch_data( p_qold );
    //print_array( ( float *) p_qold->data, 4*p_qold->set->size, "p_qold" );
//    print_array( p_q, "p_qold2" );
//    print_array( p_qold, "p_qold" );

    //assert( p_q->data[0] != 0.0f );

//  predictor/corrector update loop

    for(int k=0; k<2; k++) {

//    calculate area/timstep

      op_par_loop_adt_calc("adt_calc",cells,
                  op_arg_dat(p_x,   0,pcell, 2,"float",OP_READ ),
                  op_arg_dat(p_x,   1,pcell, 2,"float",OP_READ ),
                  op_arg_dat(p_x,   2,pcell, 2,"float",OP_READ ),
                  op_arg_dat(p_x,   3,pcell, 2,"float",OP_READ ),
                  op_arg_dat(p_q,  -1,OP_ID, 4,"float",OP_READ ),
                  op_arg_dat(p_adt,-1,OP_ID, 1,"float",OP_WRITE));
      
#ifdef DIAGNOSTIC
    if (iter==1 && k==0) {
      dump_array( p_adt, "p_adt0" );
    }
    if (iter==1 && k==1) {
      dump_array( p_adt, "p_adt1" );
    }
#endif

//    calculate flux residual

      op_par_loop_res_calc("res_calc",edges,
                  op_arg_dat(p_x,    0,pedge, 2,"float",OP_READ),
                  op_arg_dat(p_x,    1,pedge, 2,"float",OP_READ),
                  op_arg_dat(p_q,    0,pecell,4,"float",OP_READ),
                  op_arg_dat(p_q,    1,pecell,4,"float",OP_READ),
                  op_arg_dat(p_adt,  0,pecell,1,"float",OP_READ),
                  op_arg_dat(p_adt,  1,pecell,1,"float",OP_READ),
                  op_arg_dat(p_res,  0,pecell,4,"float",OP_INC ),
                  op_arg_dat(p_res,  1,pecell,4,"float",OP_INC ));

#ifdef DIAGNOSTIC
    if (iter==1 && k==0) {
      dump_array( p_res, "p_res0" );
    }
    if (iter==1 && k==1) {
      dump_array( p_res, "p_res1" );
    }
#endif

      op_par_loop_bres_calc("bres_calc",bedges,
                  op_arg_dat(p_x,     0,pbedge, 2,"float",OP_READ),
                  op_arg_dat(p_x,     1,pbedge, 2,"float",OP_READ),
                  op_arg_dat(p_q,     0,pbecell,4,"float",OP_READ),
                  op_arg_dat(p_adt,   0,pbecell,1,"float",OP_READ),
                  op_arg_dat(p_res,   0,pbecell,4,"float",OP_INC ),
                  op_arg_dat(p_bound,-1,OP_ID  ,1,"int",  OP_READ));

#ifdef DIAGNOSTIC
    if (iter==1 && k==0) {
      dump_array( p_res, "p_res_a0" );
    }
    if (iter==1 && k==0) {
      dump_array( p_res, "p_res_a1" );
    }
#endif
//    update flow field

      rms = 0.0;

      op_par_loop_update("update",cells,
                  op_arg_dat(p_qold,-1,OP_ID, 4,"float",OP_READ ),
                  op_arg_dat(p_q,   -1,OP_ID, 4,"float",OP_WRITE),
                  op_arg_dat(p_res, -1,OP_ID, 4,"float",OP_RW   ),
                  op_arg_dat(p_adt, -1,OP_ID, 1,"float",OP_READ ),
                  op_arg_gbl(&rms,1,"float",OP_INC));
    }

#ifdef DIAGNOSTIC
    if (iter==1) {
      dump_array( p_q, "p_q1" );
    }
#endif

//  print iteration history

    rms = sqrt(rms/(float) ncell);

    if (iter%100 == 0)
      printf(" %d  %10.5e \n",iter,rms);



  }

  op_timing_output();

#ifdef DIAGNOSTIC
  dump_array( p_q, "p_q" );
#endif



}

