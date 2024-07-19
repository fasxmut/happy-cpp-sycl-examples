#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <numeric>

/*
sycl::buffer
sycl::sqrt
*/

namespace kernel
{

template <std::floating_point type_xti, unsigned int dimensions>
class sqrt_kernel
{
public:
	using input_accessor_type = sycl::accessor<type_xti, dimensions, sycl::access_mode::read>;
	using output_accessor_type = sycl::accessor<type_xti, dimensions, sycl::access_mode::write>;
private:
	input_accessor_type __input;
	output_accessor_type __output;
public:
	sqrt_kernel(
		sycl::buffer<type_xti, dimensions> & input_buffer__,
		sycl::buffer<type_xti, dimensions> & output_buffer__,
		sycl::handler & handler__
	):
		__input{input_buffer__, handler__, sycl::read_only},
		__output{output_buffer__, handler__, sycl::write_only}
	{
	}
public:
	void operator()(sycl::item<dimensions> item) const
	{
		sycl::id<1> id = item.get_id();
		__output[id] = sycl::sqrt(__input[id]);
	}
};

}

int main()
{
	sycl::queue queue{sycl::gpu_selector_v};

	std::vector<float> data(7);
	std::iota(data.begin(), data.end(), 1.0f);
	auto buffer = new sycl::buffer<float, 1>{data.data(), sycl::range<1>{data.size()}};

	queue.submit(
		[&] (sycl::handler & handler)
		{
			// use the same buffer as input-buffer and output-buffer in this example.
			kernel::sqrt_kernel<float, 1u> kernel{* buffer, * buffer, handler};
			handler.parallel_for<class kn1>(
				sycl::range<1>{data.size()},
				kernel
			);
		}
	);
	queue.wait();

	delete buffer;

	for (const auto & x: data)
	{
		std::cout << ": " << x << std::endl;
	}
}

