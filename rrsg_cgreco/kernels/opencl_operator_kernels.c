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
__kernel void operator_fwd(__global float2 *out, __global float2 *in,
                       __global float2 *coils, const int NCo,
                       const int NSl, const int NScan)
{
  size_t X = get_global_size(2);
  size_t Y = get_global_size(1);

  size_t x = get_global_id(2);
  size_t y = get_global_id(1);
  size_t k = get_global_id(0);

  float2 tmp_in = 0.0f;
  float2 tmp_grad = 0.0f;
  float2 tmp_coil = 0.0f;
  float2 tmp_mul = 0.0f;


    for (int scan=0; scan<NScan; scan++)
    {
      for (int coil=0; coil < NCo; coil++)
      {
        out[scan*NCo*NSl*X*Y+coil*NSl*X*Y+k*X*Y + y*X + x] = (float2)(in[k*X*Y+ y*X + x].x*coils[coil*NSl*X*Y + k*X*Y + y*X + x].x-
                                                                      in[k*X*Y+ y*X + x].y*coils[coil*NSl*X*Y + k*X*Y + y*X + x].y,
                                                                      in[k*X*Y+ y*X + x].x*coils[coil*NSl*X*Y + k*X*Y + y*X + x].y+
                                                                      in[k*X*Y+ y*X + x].y*coils[coil*NSl*X*Y + k*X*Y + y*X + x].x);
      }
    }


}
__kernel void operator_ad(__global float2 *out, __global float2 *in,
                       __global float2 *coils, const int NCo,
                       const int NSl, const int NScan)
{
  size_t X = get_global_size(2);
  size_t Y = get_global_size(1);
  size_t x = get_global_id(2);
  size_t y = get_global_id(1);
  size_t k = get_global_id(0);


  float2 tmp_in = 0.0f;
  float2 tmp_mul = 0.0f;
  float2 conj_grad = 0.0f;
  float2 conj_coils = 0.0f;

  float2 sum = (float2) 0.0f;
  for (int scan=0; scan<NScan; scan++)
  {
  for (int coil=0; coil < NCo; coil++)
  {
    conj_coils = (float2) (coils[coil*NSl*X*Y + k*X*Y + y*X + x].x,
                                  -coils[coil*NSl*X*Y + k*X*Y + y*X + x].y);
    tmp_in = in[scan*NCo*NSl*X*Y+coil*NSl*X*Y + k*X*Y+ y*X + x];

    sum += (float2)(tmp_in.x*conj_coils.x-tmp_in.y*conj_coils.y,
                                     tmp_in.x*conj_coils.y+tmp_in.y*conj_coils.x);
  }
  }
  out[k*X*Y+y*X+x] = sum;


}
