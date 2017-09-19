#define SQRT4PI (3.544907701811)
#define PI      (3.141592653589)
#define GAMMA   (5.0/3.0)
#define SQ(x)   ((x)*(x))
#define SQ3(x,y,z) (SQ(x) + SQ(y) + SQ(z))
#define VDOT3(x1,y1,z1,x2,y2,z2) ((x1)*(x2) + (y1)*(y2) + (z1)*(z2))
#define ROEAVE(sr0,sr1,u0,u1) (((sr0)*(u0)+(sr1)*(u1))/(sr0 + sr1))
/*
	rho := mass density
	p   := momentum density
	eps := energy density
	B   := magnetic flux density
*/

float hll_select(
	  const float SL, const float SR
	, const float U0, const float U1
	, const float UHLL
)
{
	if (SL > 0) {
		return U0;
	} else if (SR < 0) {
		return U1;
	} else {
		return UHLL;
	}
}

float hll_value(
	  const float SL, const float SR
	, const float U0, const float U1
	, const float F0, const float F1
)
{
	const float UHLL = (SR*U1 - SL*U0 + F0 - F1) / (SR - SL);
	return hll_select(SL, SR, U0, U1, UHLL);
}
 
float hll_flux(
	  const float SL, const float SR
	, const float U0, const float U1
	, const float F0, const float F1
)
{
	const float FHLL = (SR*F0 - SL*F1 + SL*SR*(U1 - U0)) / (SR - SL);
	return hll_select(SL, SR, F0, F1, FHLL);
}

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
	  __global float *buf
	, __global float *buf_next
	, const size_t shape
)
{
	const int gid = get_global_id(0);
	__global float *rho = buf + shape * 0 + gid;
	__global float *p_x = buf + shape * 1 + gid;
	__global float *p_y = buf + shape * 2 + gid;
	__global float *p_z = buf + shape * 3 + gid;
	__global float *eps = buf + shape * 4 + gid;
	__global float *B_x = buf + shape * 5 + gid;
	__global float *B_y = buf + shape * 6 + gid;
	__global float *B_z = buf + shape * 7 + gid;
	__global float *rho_next = buf_next + shape * 0 + gid;
	__global float *p_x_next = buf_next + shape * 1 + gid;
	__global float *p_y_next = buf_next + shape * 2 + gid;
	__global float *p_z_next = buf_next + shape * 3 + gid;
	__global float *eps_next = buf_next + shape * 4 + gid;
	__global float *B_x_next = buf_next + shape * 5 + gid;
	__global float *B_y_next = buf_next + shape * 6 + gid;
	__global float *B_z_next = buf_next + shape * 7 + gid;

	const float eps_B[2] = { SQ3(B_x[0], B_y[0], B_z[0]) / (8.0*PI)
						   , SQ3(B_x[1], B_y[1], B_z[1]) / (8.0*PI) };
	const float v_x[2] = { p_x[0] / rho[0], p_x[1] / rho[1] };
	const float v_y[2] = { p_y[0] / rho[0], p_y[1] / rho[1] };
	const float v_z[2] = { p_z[0] / rho[0], p_z[1] / rho[1] };
	const float P[2] = { (GAMMA - 1.0) * (eps[0] - VDOT3(p_x[0], p_y[0], p_z[0], v_x[0], v_y[0], v_z[0]) / 2.0 - eps_B[0])
					   , (GAMMA - 1.0) * (eps[1] - VDOT3(p_x[1], p_y[1], p_z[1], v_x[1], v_y[1], v_z[1]) / 2.0 - eps_B[1]) };
	const float H[2] = { GAMMA * P[0] / (GAMMA - 1.0) + eps_B[0]*2.0
					   , GAMMA * P[1] / (GAMMA - 1.0) + eps_B[1]*2.0 };

	const float sqrt_rho[2] = { sqrt(rho[0]), sqrt(rho[1]) };
	const float rho_ave = sqrt_rho[0] * sqrt_rho[1];
	const float v_x_ave = ROEAVE(sqrt_rho[0], sqrt_rho[1], v_x[0], v_x[1]);
	const float H_ave = ROEAVE(sqrt_rho[0], sqrt_rho[1], H[0], H[1]);

	const float Vax2 = B_x[0] * B_x[1] / (4.0 * PI * rho_ave);
	const float Vax = sqrt(Vax2);
	const float sqrt_D = sqrt(SQ(v_x_ave) + 4.0*(GAMMA - 1.0)*(H_ave - Vax2));

	const float SR = fmax( Vax, (v_x_ave + sqrt_D) / 2.0);
	const float SL = fmin(-Vax, (v_x_ave - sqrt_D) / 2.0);

	const float p_x_flux[2] = { P[0] + eps_B[0] - SQ(B_x[0]) / (4.0 * PI)
							  , P[1] + eps_B[1] - SQ(B_x[1]) / (4.0 * PI) };
	const float p_x_hll = hll_value(SL, SR, p_x[0], p_x[1], p_x_flux[0], p_x_flux[1]);
	const float rho_hll = hll_value(SL, SR, rho[0], rho[1], 0, 0);
	const float v_x_hll = p_x_hll / rho_hll;

	const float rho_flux_hll = hll_flux(SL, SR, rho[0], rho[1], 0.0, 0.0);
	const float p_x_flux[2] = { P + epsB[0] - SQ(B_x[0]) / (4.0*PI)
							  , P + epsB[0] - SQ(B_x[1]) / (4.0*PI) };
	const float p_y_flux[2] = { -B_x[0]*B_y[0] / (4.0*PI)
							  , -B_x[1]*B_y[1] / (4.0*PI) };
	const float p_z_flux[2] = { -B_x[0]*B_z[0] / (4.0*PI)
							  , -B_x[1]*B_z[1] / (4.0*PI) };
	const float p_x_flux_hll = hll_flux(SL, SR, p_x[0], p_x[1], p_x_flux[0], p_x_flux[1]);
	const float p_y_flux_hll = hll_flux(SL, SR, p_y[0], p_y[1], p_y_flux[0], p_y_flux[1]);
	const float p_z_flux_hll = hll_flux(SL, SR, p_z[0], p_z[1], p_z_flux[0], p_z_flux[1]);
	const float eps_flux[2] = { H[0] * v_x[0] - B_x[0] * (VDOT3(v_x[0], v_y[0], v_z[0], B_x[0], B_y[0], B_z[0]) / (4.0*PI))
							  , H[1] * v_x[1] - B_x[1] * (VDOT3(v_x[1], v_y[1], v_z[1], B_x[1], B_y[1], B_z[1]) / (4.0*PI)) };
	const float eps_flux_hll = hll_flux(SL, SR, eps[0], eps[1], eps_flux[0], eps_flux[1]);
	const float B_x_flux[2] = { 0.0, 0.0 };
	const float B_y_flux[2] = { -v_y[0] * B_x[0], -v_y[1] * B_x[1] };
	const float B_z_flux[2] = { -v_z[0] * B_x[0], -v_y[1] * B_x[1] };
	const float B_x_flux_hll = hll_flux(SL, SR, B_x[0], B_x[1], B_x_flux[0], B_x_flux[1]);
	const float B_y_flux_hll = hll_flux(SL, SR, B_y[0], B_y[1], B_y_flux[0], B_y_flux[1]);
	const float B_z_flux_hll = hll_flux(SL, SR, B_z[0], B_z[1], B_z_flux[0], B_z_flux[1]);
}