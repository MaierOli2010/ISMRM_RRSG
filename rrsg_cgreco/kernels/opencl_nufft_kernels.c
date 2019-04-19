/*
Copyright 2019 Oliver Maier

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/



void AtomicAdd(volatile __global float *addr, float val) {
/*Workaround for OpenCL floating point atomic operations.
  Posted by ANCA HAMURARU ON 9 FEBRUARY 2016 WITH at
  https://streamhpc.com/blog/2016-02-09/atomic-operations-for-floats-in-opencl-improved
  */
  union {
           unsigned int u32;
           float        f32;
       } next, expected, current;
   	current.f32    = *addr;
       do {
   	   expected.f32 = current.f32;
           next.f32     = expected.f32 + val;
   		current.u32  = atomic_cmpxchg( (volatile __global unsigned int *)addr,
                               expected.u32, next.u32);
       } while( current.u32 != expected.u32 );
}


__kernel void deapo_adj(__global float2 *out, __global float2 *in, __constant float *deapo, const int dim, const float scale, const float ogf){
    size_t x = get_global_id(2);
    size_t X = get_global_size(2);
    size_t y = get_global_id(1);
    size_t Y = get_global_size(1);
    size_t k = get_global_id(0);

    size_t m = x+(int)(dim-X)/2;
    size_t M = dim;
    size_t n = y+(int)(dim-Y)/2;
    size_t N = dim;

    out[k*X*Y+y*X+x] = in[k*N*M+n*M+m] * deapo[y]* deapo[x] * scale;
}


__kernel void deapo_fwd(__global float2 *out, __global float2 *in, __constant float *deapo, const int dim, const float scale, const float ogf){
    size_t x = get_global_id(2);
    size_t X = get_global_size(2);
    size_t y = get_global_id(1);
    size_t Y = get_global_size(1);
    size_t k = get_global_id(0);

    size_t m = x+(int)(dim-X)/2;
    size_t M = dim;
    size_t n = y+(int)(dim-Y)/2;
    size_t N = dim;


    out[k*N*M+n*M+m] = in[k*X*Y+y*X+x] * deapo[y]* deapo[x] * scale;
  }

  __kernel void zero_tmp(__global float2 *tmp)
  {
    size_t x = get_global_id(0);
    tmp[x] = 0.0f;
}


__kernel void grid_lut(__global float *sg, __global float2 *s, __global float2 *kpos, const int gridsize, const float kwidth, __global float *dcf, __constant float* kerneltable, const int nkernelpts ){
/* Gridding function that uses a lookup table for a circularyl
  symmetric convolution kernel, with linear interpolation.
  Original Function by B. Hargreaves
  Adapted and ported to OpenCL by O. Maier*/
    size_t k = get_global_id(2);
    size_t kDim = get_global_size(2);
    size_t n = get_global_id(1);
    size_t NDim = get_global_size(1);
    size_t scan = get_global_id(0);

    int ixmin, ixmax, iymin, iymax, gridcenter, gptr_cinc, kernelind,indx,indy;
    float kx, ky;
    float fracind, dkx, dky, dk, fracdk, kern;
    gridcenter = gridsize/2;


    float* ptr, pti;
    float2 kdat = s[k+kDim*n+kDim*NDim*scan]*(float2)(dcf[k],dcf[k]);



    kx = (kpos[k+kDim*scan]).s0;
    ky = (kpos[k+kDim*scan]).s1;


    ixmin =  (int)((kx-kwidth)*gridsize +gridcenter);
    ixmax = (int)((kx+kwidth)*gridsize +gridcenter)+1;
    iymin = (int)((ky-kwidth)*gridsize +gridcenter);
    iymax =  (int)((ky+kwidth)*gridsize +gridcenter)+1;

    	for (int gcount1 = ixmin; gcount1 <= ixmax; gcount1++)
    {
      	dkx = (float)(gcount1-gridcenter) / (float)gridsize  - kx;
      	for (int gcount2 = iymin; gcount2 <= iymax; gcount2++)
      	{
        	dky = (float)(gcount2-gridcenter) / (float)gridsize - ky;
        dk = sqrt(dkx*dkx+dky*dky);
        if (dk < kwidth)
        {
          fracind = dk/kwidth*(float)(nkernelpts-1);
          kernelind = (int)fracind;
          fracdk = fracind-(float)kernelind;

          kern = kerneltable[kernelind]*(1-fracdk)+
          kerneltable[kernelind+1]*fracdk;
          indx = gcount1;
          indy = gcount2;
          if (gcount1 < 0) {indx+=gridsize;indy=gridsize-indy;}
          if (gcount1 >= gridsize) {indx-=gridsize;indy=gridsize-indy;}
          if (gcount2 < 0) {indy+=gridsize;indx=gridsize-indx;}
          if (gcount2 >= gridsize) {indy-=gridsize;indx=gridsize-indx;}
          AtomicAdd(&(sg[2*(indx*gridsize+indy+(gridsize*gridsize)*n+(gridsize*gridsize)*NDim*scan)]),(kern * kdat.s0));
          AtomicAdd(&(sg[2*(indx*gridsize+indy+(gridsize*gridsize)*n+(gridsize*gridsize)*NDim*scan)+1]),(kern * kdat.s1));
        }
      }
    }
}


__kernel void invgrid_lut(__global float2 *s, __global float2 *sg, __global float2 *kpos, const int gridsize, const float kwidth, __global float *dcf, __constant float* kerneltable, const int nkernelpts ){
/* Sampling the non uniform grid using a lookup table for a circularyl
  symmetric convolution kernel, with linear interpolation.
  Original Function by B. Hargreaves
  Adapted and ported to OpenCL by O. Maier*/
    size_t k = get_global_id(2);
    size_t kDim = get_global_size(2);
    size_t n = get_global_id(1);
    size_t NDim = get_global_size(1);
    size_t scan = get_global_id(0);

    int ixmin, ixmax, iymin, iymax, gridcenter, gptr_cinc, kernelind, indx,indy;
    float kx, ky;
    float fracind, dkx, dky, dk, fracdk, kern;
    gridcenter = gridsize/2;

    float2 tmp_dat = 0.0;


    kx = (kpos[k+kDim*scan]).s0;
    ky = (kpos[k+kDim*scan]).s1;

    ixmin =  (int)((kx-kwidth)*gridsize +gridcenter);
    ixmax = (int)((kx+kwidth)*gridsize +gridcenter)+1;
    iymin = (int)((ky-kwidth)*gridsize +gridcenter);
    iymax =  (int)((ky+kwidth)*gridsize +gridcenter)+1;

    	for (int gcount1 = ixmin; gcount1 <= ixmax; gcount1++)
    	{
      	dkx = (float)(gcount1-gridcenter) / (float)gridsize  - kx;
      	for (int gcount2 = iymin; gcount2 <= iymax; gcount2++)
      	{
        	dky = (float)(gcount2-gridcenter) / (float)gridsize - ky;
        dk = sqrt(dkx*dkx+dky*dky);
        if (dk < kwidth)
        {
          fracind = dk/kwidth*(float)(nkernelpts-1);
          kernelind = (int)fracind;
          fracdk = fracind-(float)kernelind;
          kern = kerneltable[kernelind]*(1-fracdk)+
          kerneltable[kernelind+1]*fracdk;
          indx = gcount1;
          indy = gcount2;
          if (gcount1 < 0) {indx+=gridsize;indy=gridsize-indy;}
          if (gcount1 >= gridsize) {indx-=gridsize;indy=gridsize-indy;}
          if (gcount2 < 0) {indy+=gridsize;indx=gridsize-indx;}
          if (gcount2 >= gridsize) {indy-=gridsize;indx=gridsize-indx;}
          tmp_dat += (float2)(kern,kern)*sg[indx*gridsize+indy+(gridsize*gridsize)*n+(gridsize*gridsize)*NDim*scan];
        }
      }
    	}
    	s[k+kDim*n+kDim*NDim*scan]= tmp_dat*(float2)(dcf[k],dcf[k]);
}


__kernel void fftshift(__global float2* ksp, __global float *check){
    size_t x = get_global_id(2);
    size_t dimX = get_global_size(2);
    size_t y = get_global_id(1);
    size_t dimY = get_global_size(1);
    size_t n = get_global_id(0);

    ksp[x+dimX*y+dimX*dimY*n] = ksp[x+dimX*y+dimX*dimY*n]*check[x]*check[y];
}
