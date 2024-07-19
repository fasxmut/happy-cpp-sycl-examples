#include <sycl/sycl.hpp>
#include <vector>
#include <iomanip>

namespace gpu
{

class range_info
{
public:
	constexpr static const int
		gdimy = 4,
		gdimx = 4,
		ldimy = 2,
		ldimx = 2,
		gsize = gdimy * gdimx,
		lm_offset = 9
	;
	static_assert(gdimy == gdimx);
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
		sycl::buffer<value_type, 2> & m0__,
		sycl::buffer<value_type, 2> & m1__,
		sycl::buffer<value_type, 2> & m2__,
		const sycl::range<3> & lm_range__,
		sycl::handler & handler__
	):
		__matrix0{m0__, handler__, sycl::read_only},
		__matrix1{m1__, handler__, sycl::read_only},
		__matrix2{m2__, handler__, sycl::write_only},
		__lm{lm_range__, handler__}
	{
	}
public:
	void operator()(sycl::nd_item<2> item) const
	{
		auto gidy = item.get_global_id(0);
		auto gidx = item.get_global_id(1);
		auto lidy = item.get_local_id(0);
		auto lidx = item.get_local_id(1);

		auto gsizey = item.get_global_range()[0];
		auto gsizex = item.get_global_range()[1];

		if (gsizey != gsizex)
		{
			__matrix2[gidy][gidx] = 999;	// indicate error
			return;
		}

		// else

		// copy the row from first matrix to lm
		for (int i=0; i<gsizex; ++i)
			__lm[lidy][lidx][i] = __matrix0[gidy][i];

		// copy the column from second matrix to lm
		for (int j=0; j<gsizey; ++j)
			__lm[lidy][lidx][j+gsizex] = __matrix1[j][gidx];

		// alias
		value_type & sum = __lm[lidy][lidx][0 + gsizey + gsizex];
		// Init the last value of lm to 0.
		sum = 0;

		// synchronize now
		sycl::group_barrier(item.get_group(), sycl::memory_scope::work_group);

		// do multiplication addition
		for (int i0=0; i0<gsizex; ++i0)
		{
			int i1 = i0 + gsizex;
			sum += __lm[lidy][lidx][i0] * __lm[lidy][lidx][i1];
		}

		// synchronize now
		sycl::group_barrier(item.get_group(), sycl::memory_scope::work_group);

		// copy result from local memory to host memory
		__matrix2[gidy][gidx] = sum;
	}
};

}	// namespace gpu

int main()
{
	constexpr auto info = gpu::range_info{};
	sycl::queue queue{sycl::gpu_selector_v};
	using value_type = int;

	auto matrix0 = std::vector<value_type>{
		1,2,3,4,
		3,2,-1,-2,
		-2,2,3,2,
		4,2,-3,4
	};
	auto m0_buff = sycl::buffer<value_type, 2>{matrix0.data(), sycl::range<2>{info.gdimy, info.gdimx}};

	auto matrix1 = std::vector<value_type>{
		2,1,-2,-3,
		3,2,4,5,
		2,-2,3,4,
		-2,-3,-3,-4
	};
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
			handler.parallel_for(
				sycl::nd_range<2>{
					sycl::range<2>{info.gdimy, info.gdimx},
					sycl::range<2>{info.ldimy, info.ldimx},
				},
				kernel
			);
		}
	);
	
	auto print = [&] (const auto data)
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

	print(matrix0);

	std::cout << "x\n";

	print(matrix1);

	std::cout << "=\n";

	auto host_access = m2_buff.get_host_access();
	// print result
	for (int j=0; j<info.gdimy; ++j)
	{
		for (int i=0; i<info.gdimx; ++i)
			std::cout << std::setw(5) << host_access[j][i];
		std::cout << std::endl;
	}
	std::cout << std::endl;
}

// output:
/*
    1    2    3    4
    3    2   -1   -2
   -2    2    3    2
    4    2   -3    4

x
    2    1   -2   -3
    3    2    4    5
    2   -2    3    4
   -2   -3   -3   -4

=
    6  -13    3    3
   14   15    5    5
    4  -10   15   20
    0    2  -21  -30

*/

