#include <sycl/sycl.hpp>
#include <iostream>
#include <iomanip>
#include <array>

namespace gpu
{

class range_info
{
public:
	const int
		gdimy,
		gdimx,
		ldimy,
		ldimx,
		gsize,
		lm_offset = 3
	;
};

template <typename value_type>
class kernel2d_class
{
private:
	sycl::accessor<value_type, 2, sycl::access_mode::read> __matrix0;
	sycl::accessor<value_type, 2, sycl::access_mode::read> __matrix1;
	sycl::accessor<value_type, 2, sycl::access_mode::write> __matrix2;
	sycl::local_accessor<value_type, 3> __lm;
public:
	kernel2d_class(
		sycl::buffer<value_type, 2> & matrix0__,
		sycl::buffer<value_type, 2> & matrix1__,
		sycl::buffer<value_type, 2> & matrix2__,
		const sycl::range<3> & lm_range__,
		sycl::handler & handler__
	):
		__matrix0{matrix0__, handler__, sycl::read_only},
		__matrix1{matrix1__, handler__, sycl::read_only},
		__matrix2{matrix2__, handler__, sycl::write_only},
		__lm{lm_range__, handler__}
	{
	}
public:
	void operator()(sycl::nd_item<2> item) const
	{
		auto gid_j = item.get_global_id(0);
		auto gid_i = item.get_global_id(1);
		auto lid_j = item.get_local_id(0);
		auto lid_i = item.get_local_id(1);

		// name shorter alias
		const value_type & m0_value = __matrix0[gid_j][gid_i];
		const value_type & m1_value = __matrix1[gid_j][gid_i];
		value_type & m2_value = __matrix2[gid_j][gid_i];

		value_type & lm0 = __lm[lid_j][lid_i][0];
		value_type & lm1 = __lm[lid_j][lid_i][1];
		value_type & lm2 = __lm[lid_j][lid_i][2];

	// Initialize local memory
		lm0 = 0;
		lm1 = 0;
		lm2 = 0;
		// synchronize with barrier
		sycl::group_barrier(item.get_group(), sycl::memory_scope::work_group);

	// copy to local memory
		lm0 = m0_value;
		sycl::group_barrier(item.get_group(), sycl::memory_scope::work_group);

		lm1 = m1_value;
		sycl::group_barrier(item.get_group(), sycl::memory_scope::work_group);

	// addition
		lm2 = lm0 + lm1;

	// copy from local memory to global memory
		m2_value = lm2;
	}
};

}	// namespace gpu

int main()
{
	sycl::queue queue{sycl::gpu_selector_v};
	using value_type = float;
	constexpr auto info = gpu::range_info{4, 4, 2, 2, 4*4};
	std::vector<value_type> matrix0{
		1,2,3,4,
		3,2,4,2,
		-1,-3,-2,1,
		7,8,4,-3
	};
	std::vector<value_type> matrix1{
		3,2,-7,5,
		2,-3,-5,1,
		4,5,7,-2,
		9,11,-7,-8
	};
	auto m0_buff = sycl::buffer<value_type, 2>{matrix0.data(), sycl::range<2>{info.gdimy, info.gdimx}};
	auto m1_buff = sycl::buffer<value_type, 2>{matrix1.data(), sycl::range<2>{info.gdimy, info.gdimx}};
	auto m2_buff = sycl::buffer<value_type, 2>{sycl::range<2>{info.gdimy, info.gdimx}};

	queue.submit(
		[&] (sycl::handler & handler)
		{
			auto kernel = gpu::kernel2d_class{
				m0_buff,
				m1_buff,
				m2_buff,
				sycl::range<3>{info.ldimy, info.ldimx, info.lm_offset},
				handler
			};
			handler.parallel_for<class name1>(
				sycl::nd_range<2>{
					sycl::range<2>{info.gdimy, info.gdimx},
					sycl::range<2>{info.ldimy, info.ldimx}
				},
				kernel
			);
		}
	);

	auto print = [] (const auto data, const gpu::range_info & info)
	{
		for (int j=0; j<info.gdimy; ++j)
		{
			for (int i=0; i<info.gdimx; ++i)
			{
				std::cout << std::setw(5) << data[j*info.gdimx+i];
			}
			std::cout << std::endl;
		}
		std::cout << std::endl;
	};

	print(matrix0, info);
	std::cout << "+\n";
	print(matrix1, info);
	std::cout << "=\n";

	auto matrix2_accessor = m2_buff.get_host_access();

	for (int j=0; j<info.gdimy; ++j)
	{
		for (int i=0; i<info.gdimx; ++i)
		{
			std::cout << std::setw(5) << matrix2_accessor[j][i];
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;
}

// output:
/*
    1    2    3    4
    3    2    4    2
   -1   -3   -2    1
    7    8    4   -3

+
    3    2   -7    5
    2   -3   -5    1
    4    5    7   -2
    9   11   -7   -8

=
    4    4   -4    9
    5   -1   -1    3
    3    2    5   -1
   16   19   -3  -11

*/
