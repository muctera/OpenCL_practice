#define SQRT4PI (3.544907701811)
#define PI      (3.141592653589)
#define GAMMA   (5.0/3.0)
#define SQ(x)   (x*x)
#define SQ3(x,y,z) (SQ(x) + SQ(y) + SQ(z))

float P_to_eps(
	  const float P
	, const float rho
	, const float p2
	, const float B2
)
{
	return P/(GAMMA-1.0) + p2/(2.0*rho) + B2/(8.0*PI);
}

__kernel void initialize(
	  __global float *buf
	, const size_t shape
)
{
	__global float *rho = buf + shape * 0;
	__global float *p_x = buf + shape * 1;
	__global float *p_y = buf + shape * 2;
	__global float *p_z = buf + shape * 3;
	__global float *eps = buf + shape * 4;
	__global float *B_x = buf + shape * 5;
	__global float *B_y = buf + shape * 6;
	__global float *B_z = buf + shape * 7;

	const int gid = get_global_id(0);
	bool is_left = gid < (float)shape * 0.5;

	rho[gid] = is_left ? 1.0 : 0.125;
	p_x[gid] = is_left ? 0.0 : 0.0;
	p_y[gid] = is_left ? 0.0 : 0.0;
	p_z[gid] = is_left ? 0.0 : 0.0;
	B_x[gid] = is_left ? 0.75 * SQRT4PI : 0.75 * SQRT4PI;
	B_y[gid] = is_left ? SQRT4PI : -SQRT4PI;
	B_z[gid] = is_left ? 0.0 : 0.0;

	const float p2 = SQ3(p_x[gid], p_y[gid], p_z[gid]);
	const float B2 = SQ3(B_x[gid], B_y[gid], B_z[gid]);
	const float P = is_left ? 1.0 : 0.1;
	eps[gid] = P_to_eps(P, rho[gid], p2, B2);
}

__kernel void nextstep(
	  __global *buf
	, __global *buf_next
)
{
	__global float *rho = buf + shape * 0;
	__global float *p_x = buf + shape * 1;
	__global float *p_y = buf + shape * 2;
	__global float *p_z = buf + shape * 3;
	__global float *eps = buf + shape * 4;
	__global float *B_x = buf + shape * 5;
	__global float *B_y = buf + shape * 6;
	__global float *B_z = buf + shape * 7;
	__global float *rho_next = buf_next + shape * 0;
	__global float *p_x_next = buf_next + shape * 1;
	__global float *p_y_next = buf_next + shape * 2;
	__global float *p_z_next = buf_next + shape * 3;
	__global float *eps_next = buf_next + shape * 4;
	__global float *B_x_next = buf_next + shape * 5;
	__global float *B_y_next = buf_next + shape * 6;
	__global float *B_z_next = buf_next + shape * 7;


}