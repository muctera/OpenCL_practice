#include <iostream>
#include <fstream>
#define CL_USE_DEPRECATED_OPENCL_2_0_APIS
#include <CL/cl.hpp>

std::string fetch_Program(int &argc, char ** &argv)
{
	argc--;
	argv++;
	if (argc == 0) {
		throw;
	}

	std::cerr << argv[0] << std::endl;

	std::ifstream filestream(argv[0]);
	std::string filestring(
		(std::istreambuf_iterator<char>(filestream)),
		std::istreambuf_iterator<char>()
	);

	return filestring;
}
     
int main(int argc, char *argv[])
{
	cl::Context context(CL_DEVICE_TYPE_GPU);
    std::vector<cl::Device> gpu_list = context.getInfo<CL_CONTEXT_DEVICES>();

	cl::Program program(context, fetch_Program(argc, argv));
	program.build(gpu_list, "-cl-std=CL2.0 -cl-single-precision-constant");
	std::cerr << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(gpu_list[0]) << std::endl;
	cl::CommandQueue queue(context, gpu_list[0]);


	const size_t edge = 1;
	const size_t shape = 512 + edge;
	const size_t variable_num = 8;
	std::vector<float> value(shape*variable_num);

	cl::Buffer buf0(context, CL_MEM_READ_WRITE, shape*variable_num * sizeof(float));
	cl::Buffer buf1(context, CL_MEM_READ_WRITE, shape*variable_num * sizeof(float));

	cl::Kernel initialize(program, "initialize");

	initialize.setArg(0, buf0);
	initialize.setArg(1, shape);
	queue.enqueueNDRangeKernel(initialize, cl::NDRange(0), cl::NDRange(shape));

	initialize.setArg(0, buf1);
	initialize.setArg(1, shape);
	queue.enqueueNDRangeKernel(initialize, cl::NDRange(0), cl::NDRange(shape));

	queue.enqueueReadBuffer(buf0, CL_TRUE, 0, shape*variable_num * sizeof(float), value.data());
	for (size_t i = 0; i < shape; i++) {
		std::cout << i << " " << value.at(i + shape*4) << std::endl;
	}

	cl::Kernel nextstep(program, "nextstep");

	nextstep.setArg(0, buf0);
	nextstep.setArg(1, buf1);
	nextstep.setArg(2, shape);
	queue.enqueueNDRangeKernel(nextstep, cl::NDRange(0), cl::NDRange(shape - edge));

	queue.enqueueReadBuffer(buf1, CL_TRUE, 0, shape*variable_num * sizeof(float), value.data());
	for (size_t i = 0; i < shape; i++) {
		std::cout << i << " " << value.at(i + shape*4) << std::endl;
	}

	nextstep.setArg(0, buf1);
	nextstep.setArg(1, buf0);
	nextstep.setArg(2, shape);
	queue.enqueueNDRangeKernel(nextstep, cl::NDRange(0), cl::NDRange(shape - edge));

	queue.enqueueReadBuffer(buf0, CL_TRUE, 0, shape*variable_num * sizeof(float), value.data());
	for (size_t i = 0; i < shape; i++) {
		std::cout << i << " " << value.at(i + shape*4) << std::endl;
	}
#ifdef _WIN32
	system("pause");
#endif

    return 0;
}
