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

	cl::Program program(context, fetch_Program(argc, argv), true);
	cl::CommandQueue queue(context, gpu_list[0]);

	char string[20] = { 0 };
	cl::Buffer buffer(context, CL_MEM_WRITE_ONLY, sizeof(string));

	cl::Kernel kernel(program, "hello");
	kernel.setArg(0, buffer);

	queue.enqueueTask(kernel);
	queue.enqueueReadBuffer(buffer, CL_TRUE, 0, sizeof(string), string);

	puts(string);
#ifdef _WIN32
	system("pause");
#endif

    return 0;
}
