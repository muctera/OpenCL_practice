#include <iostream>
#define CL_USE_DEPRECATED_OPENCL_2_0_APIS
#include <CL/cl.hpp>

cl::Device get_GPU()
{
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    for (auto &platform : platforms) {
        std::cout << static_cast<std::string>(platform.getInfo<CL_PLATFORM_NAME>()) << std::endl;

        std::vector<cl::Device> gpu_devices;
        platform.getDevices(CL_DEVICE_TYPE_GPU, &gpu_devices);
        for (auto &device : gpu_devices) {
            std::cout << "\t" << static_cast<std::string>(device.getInfo<CL_DEVICE_NAME>()) << std::endl;
        }
    }
}
     
int main(int argc, char *argv[])
{

    cl::Device gpu = get_GPU();

    return 0;
}
