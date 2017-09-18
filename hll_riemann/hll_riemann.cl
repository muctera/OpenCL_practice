#define SQRT4PI (3.544907701811)
#define PI      (3.141592653589)
#define GAMMA   (5.0/3.0)
#define SQ(x)   (x*x)
#define SQ3(x,y,z) (SQ(x) + SQ(y) + SQ(z))

float pressure_to_energy(const float pressure
		   , const float density
		   , const float momentum2
		   , const float magnetflux2
		   )
{
	return pressure/(GAMMA-1.0) + momentum2/(2.0*density) + magnetflux2/(8.0*PI);
}

__kernel void initialize(__global float *buf
					   , const size_t shape
					   )
{
	__global float *density      = buf + shape * 0;
	__global float *momentum_x   = buf + shape * 1;
	__global float *momentum_y   = buf + shape * 2;
	__global float *momentum_z   = buf + shape * 3;
	__global float *energy       = buf + shape * 4;
	__global float *magnetflux_x = buf + shape * 5;
	__global float *magnetflux_y = buf + shape * 6;
	__global float *magnetflux_z = buf + shape * 7;

	const int gid = get_global_id(0);
	bool is_left = gid < (float)shape * 0.5;

	density[gid]      = is_left ? 1.0 : 0.125;
	momentum_x[gid]   = is_left ? 0.0 : 0.0;
	momentum_y[gid]   = is_left ? 0.0 : 0.0;
	momentum_z[gid]   = is_left ? 0.0 : 0.0;
	magnetflux_x[gid] = is_left ? 0.75 * SQRT4PI : 0.75 * SQRT4PI;
	magnetflux_y[gid] = is_left ? SQRT4PI : -SQRT4PI;
	magnetflux_z[gid] = is_left ? 0.0 : 0.0;

	const float momentum2 = SQ3(momentum_x[gid], momentum_y[gid], momentum_z[gid]);
	const float magnetflux2 = SQ3(magnetflux_x[gid], magnetflux_y[gid], magnetflux_z[gid]);
	const float pressure = is_left ? 1.0 : 0.1;
	energy[gid] = pressure_to_energy(pressure, density[gid], momentum2, magnetflux2);
}

