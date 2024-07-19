#include <sycl/sycl.hpp>
#include <vector>
#include <numeric>
#include <algorithm>

/*
sycl::buffer:
	Use an input buffer and an output buffer.
	Then get the data using buffer.get_host_access()
*/

template <std::floating_point type_xti, unsigned int dimensions>
class kernel_class
{
private:
	sycl::accessor<type_xti, dimensions, sycl::access_mode::read> __input;
	sycl::accessor<type_xti, dimensions, sycl::access_mode::write> __output;
public:
	kernel_class(
		sycl::buffer<type_xti, dimensions> & in_buffer__,
		sycl::buffer<type_xti, dimensions> & out_buffer__,
		sycl::handler & handler__
	):
		__input{in_buffer__, handler__, sycl::read_only},
		__output{out_buffer__, handler__, sycl::write_only}
	{
	}
public:
	// kernel
	void operator()(sycl::item<dimensions> item) const
	{
		sycl::id<1> id = item.get_id();
		__output[id] = sycl::sqrt(__input[id]);
	}
};

int main()
{
	sycl::queue queue{sycl::gpu_selector_v};
	std::vector<float> input(32);
	std::iota(input.begin(), input.end(), 1.0f);
	auto in_buffer = sycl::buffer<float, 1>{input.data(), sycl::range<1>{input.size()}};
	auto out_buffer = sycl::buffer<float, 1>{sycl::range<1>{input.size()}};
	queue.submit(
		[&] (sycl::handler & handler)
		{
			kernel_class<float, 1u> kernel{in_buffer, out_buffer, handler};
			handler.parallel_for(
				sycl::range<1>{input.size()},
				kernel
			);
		}
	);

	// out_buffer.get_host_access();
	auto host_accessor = out_buffer.get_host_access();
	std::copy(host_accessor.begin(), host_accessor.end(), std::ostream_iterator<float>(std::cout, "\n"));
}

