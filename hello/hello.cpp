#include <iostream>
#define CL_USE_DEPRECATED_OPENCL_2_0_APIS
#include <CL/cl.hpp>

std::vector<cl::Device> get_Device_List(const cl_device_type type)
{
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    std::vector<cl::Device> device_list;
    for (auto &platform : platforms) {
        std::vector<cl::Device> devices;
        platform.getDevices(type, &devices);
        for (auto &device : devices) {
            device_list.push_back(device);
        }
    }
    return device_list;
}
     
int main()
{

    cl::Device gpu = get_Device_List(CL_DEVICE_TYPE_GPU)[0];
    cl::Platform platform = gpu.getInfo<CL_DEVICE_PLATFORM>();

    std::cout << static_cast<std::string>(platform.getInfo<CL_PLATFORM_NAME>()) << std::endl;
    std::cout << static_cast<std::string>(gpu.getInfo<CL_DEVICE_NAME>()) << std::endl;

    return 0;
}
