#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <numeric>
#include <iomanip>

// partition work group by sycl nd-range

template <std::floating_point value_type>
class kernel2d_class
{
private:
	sycl::accessor<value_type, 2, sycl::access_mode::read> __input;
	sycl::accessor<value_type, 2, sycl::access_mode::write> __output;
public:
	kernel2d_class(
		sycl::buffer<value_type, 2> & in_buffer__,
		sycl::buffer<value_type, 2> & out_buffer__,
		sycl::handler & handler__
	):
		__input{in_buffer__, handler__, sycl::read_only},
		__output{out_buffer__, handler__, sycl::write_only}
	{
	}
public:
	void operator()(sycl::nd_item<2> item__) const
	{
		const sycl::id<1> idy = item__.get_global_id(0);
		const sycl::id<1> idx = item__.get_global_id(1);
		const sycl::id<1> lidy = item__.get_local_id(0);
		const sycl::id<1> lidx = item__.get_local_id(1);
		__output[idy][idx] = sycl::sqrt(__input[idy][idx]);
	}
};

int main()
{
	constexpr int
		sizey = 16, sizex =8,		// global size: 16 x 8
		lsizey = 4, lsizex = 4		// work group size: 4 x 4 (local size)
	;
	// Requires:
	//		sizey ==  N * lsizey 
	//		sizex == N * lsizex

	sycl::queue queue{sycl::gpu_selector_v};
	std::vector<double> input(sizey*sizex);
	std::iota(input.begin(), input.end(), 1.0);
	auto in_buffer = sycl::buffer<double, 2>{input.data(), sycl::range<2>{sizey, sizex}};
	auto out_buffer = sycl::buffer<double, 2>{sycl::range<2>{sizey, sizex}};
	queue.submit(
		[&] (sycl::handler & handler)
		{
			kernel2d_class kernel{in_buffer, out_buffer, handler};
			handler.parallel_for(
				sycl::nd_range<2>{
					sycl::range<2>{sizey, sizex},
					sycl::range<2>{lsizey, lsizex}	// Each work group size is (lsizey * lsizex)
				},
				kernel
			);
		}
	);

	auto host_accessor = out_buffer.get_host_access();

	for (int j=0; j<lsizey; ++j)
	{
		for (int i=0; i<lsizex; ++i)
		{
			std::cout << std::setw(10) << host_accessor[j][i] << ' ';
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;
}

// output:
/*
         1    1.41421    1.73205          2 
         3    3.16228    3.31662     3.4641 
   4.12311    4.24264     4.3589    4.47214 
         5    5.09902    5.19615     5.2915 

*/

