
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
