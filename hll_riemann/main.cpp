#include<H5Cpp.h>
#include <iostream>
#include <fstream>
#define CL_USE_DEPRECATED_OPENCL_2_0_APIS
#include <CL/cl.hpp>
#include<boost/format.hpp>

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

std::string outputfilename(int &argc, char ** &argv)
{
	argc--;
	argv++;
	if (argc == 0) {
		throw;
	}

	std::cerr << argv[0] << std::endl;

	return (boost::format("%s\\outfile.hdf5") % argv[0]).str();
}

void write_file(
	  H5::H5File &outputfile
	, int &group_id
	, const size_t &shape
	, const std::vector<std::string> &valname
	, std::vector<float> &value
	)
{
	const size_t variable_num = valname.size();
	std::vector<H5::DataSet> dataset(variable_num);
	H5::DataSpace space(1, &shape);
	H5::Group group = outputfile.createGroup((boost::format("%d") % group_id++).str());
	for (size_t i = 0; i < variable_num; i++) {
		dataset[i] = group.createDataSet(valname[i], H5::PredType::NATIVE_FLOAT, space);
		dataset[i].write(value.data() + shape * i, H5::PredType::NATIVE_FLOAT);
		dataset[i].close();
	}
	group.close();
	space.close();
}
     
int main(int argc, char *argv[])
{
	cl::Context context(CL_DEVICE_TYPE_GPU);
    std::vector<cl::Device> gpu_list = context.getInfo<CL_CONTEXT_DEVICES>();

	cl::Program program(context, fetch_Program(argc, argv));
	program.build(gpu_list, "-cl-std=CL2.0 -cl-single-precision-constant");
	std::cerr << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(gpu_list[0]) << std::endl;
	cl::CommandQueue queue(context, gpu_list[0]);


	const size_t halo = 1;
	const size_t shape = 512;
	const std::vector<std::string> valname{ "rho", "p_x", "p_y", "p_z", "eps", "B_x", "B_y", "B_z" };
	const size_t variable_num = valname.size();
	std::vector<float> value(shape*variable_num);

	cl::Buffer value_buf(context, CL_MEM_READ_WRITE, shape*variable_num * sizeof(float));
	cl::Buffer flux_buf(context, CL_MEM_READ_WRITE, shape*variable_num * sizeof(float));

	cl::Kernel initialize(program, "initialize");
	initialize.setArg(0, value_buf);
	initialize.setArg(1, shape);

	cl::Kernel calc_flux(program, "calc_flux");
	calc_flux.setArg(0, value_buf);
	calc_flux.setArg(1, flux_buf);
	calc_flux.setArg(2, shape);

	cl::Kernel nextstep(program, "nextstep");
	nextstep.setArg(0, flux_buf);
	nextstep.setArg(1, value_buf);
	nextstep.setArg(2, shape);

	
	int group_id = 0;
	H5::H5File outputfile(outputfilename(argc, argv), H5F_ACC_TRUNC);

	queue.enqueueNDRangeKernel(initialize, cl::NDRange(0), cl::NDRange(shape));
	queue.enqueueReadBuffer(value_buf, CL_TRUE, 0, shape*variable_num * sizeof(float), value.data());
	write_file(outputfile, group_id, shape, valname, value);

	for (int i = 1; i <= 1024; i++) {
		queue.enqueueNDRangeKernel(calc_flux, cl::NDRange(0), cl::NDRange(shape - halo));
		queue.enqueueNDRangeKernel(nextstep, cl::NDRange(halo), cl::NDRange(shape - 2*halo));
		if (i % 32 == 0) {
			queue.enqueueReadBuffer(value_buf, CL_TRUE, 0, shape*variable_num * sizeof(float), value.data());
			write_file(outputfile, group_id, shape, valname, value);
		}
	}

	outputfile.close();


#ifdef _WIN32
	system("pause");
#endif

    return 0;
}
