#include <sycl/sycl.hpp>
#include <vector>
#include <numeric>
#include <iomanip>

// sycl::local_accessor:
/*
	The sycl::local_accessor class allocates device local memory and provides
	access to this memory from within a SYCL kernel function. 
*/

template <std::floating_point value_type>
class kernel2d_class
{
private:
	sycl::accessor<value_type, 2, sycl::access_mode::read> __input;
	sycl::accessor<value_type, 2, sycl::access_mode::write> __output;
	sycl::local_accessor<value_type, 3> __lm;	// shared local memory
public:
	kernel2d_class(
		sycl::buffer<value_type, 2> & in_buffer__,
		sycl::buffer<value_type, 2> & out_buffer__,
		const sycl::range<3> & lm_range__,
		sycl::handler & handler__
	):
		__input{in_buffer__, handler__, sycl::read_only},
		__output{out_buffer__, handler__, sycl::write_only},
		__lm{lm_range__, handler__}
	{
	}
public:
	void operator()(sycl::nd_item<2> item__) const
	{
	// get access index
		// global
		auto idy = item__.get_global_id(0);
		auto idx = item__.get_global_id(1);
		// local (work group)
		auto lidy = item__.get_local_id(0);
		auto lidx = item__.get_local_id(1);

/*
	// get range size
		// global
		auto sizey = item__.get_global_range(0);
		auto sizex = item__.get_global_range(1);
		// local (work group)
		auto lsizey = item__.get_local_range(0);
		auto lsizex = item__.get_local_range(1);
*/

	// copy item value from host to local memory 0
		__lm[lidy][lidx][0] = __input[idy][idx];

	// copy item value from local memory 0 to local memory 1
		__lm[lidy][lidx][1] = __lm[lidy][lidx][0];

	// Calculates local memory 2 from local memory 1
		__lm[lidy][lidx][2] = sycl::sqrt(__lm[lidy][lidx][1]);

	// copy result from local memory 2 to global memory
		__output[idy][idx] = __lm[lidy][lidx][2];
	}
};

int main()
{
	sycl::queue queue{sycl::gpu_selector_v};
	constexpr int
		sizey = 24, sizex = 8,		// global size: 24 x 8
		size = sizey * sizex,
		lsizey = 6, lsizex = 4,		// work group size: 6 x 4	(local size)
		lm_offset = 3		// local memory size is: 6 x 4 x 3, so each work item consumes 3 memory units.
	;

	using value_type = double;

	std::vector<value_type> input(size);
	std::iota(input.begin(), input.end(), 1.0);
	sycl::buffer<value_type, 2> in_buffer{input.data(), sycl::range<2>{sizey, sizex}};
	sycl::buffer<value_type, 2> out_buffer{sycl::range<2>{sizey, sizex}};

	queue.submit(
		[&] (sycl::handler & handler)
		{
			auto kernel = kernel2d_class{
				in_buffer,
				out_buffer,
				sycl::range<3>{lsizey, lsizex, lm_offset},
				handler
			};
			handler.parallel_for<class kn1>(
				sycl::nd_range<2>{
					sycl::range<2>{sizey, sizex},
					sycl::range<2>{lsizey, lsizex}
				},
				kernel
			);
		}
	);
	queue.wait();

	auto host_access = out_buffer.get_host_access();

	// print
	for (int j=0; j<sizey; ++j)
	{
		for (int i=0; i<sizex; ++i)
		{
			std::cout << std::setw(10) << std::setprecision(4) << host_access[j][i];
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;
}
// output:
/*
         1     1.414     1.732         2     2.236     2.449     2.646     2.828
         3     3.162     3.317     3.464     3.606     3.742     3.873         4
     4.123     4.243     4.359     4.472     4.583      4.69     4.796     4.899
         5     5.099     5.196     5.292     5.385     5.477     5.568     5.657
     5.745     5.831     5.916         6     6.083     6.164     6.245     6.325
     6.403     6.481     6.557     6.633     6.708     6.782     6.856     6.928
         7     7.071     7.141     7.211      7.28     7.348     7.416     7.483
      7.55     7.616     7.681     7.746      7.81     7.874     7.937         8
     8.062     8.124     8.185     8.246     8.307     8.367     8.426     8.485
     8.544     8.602      8.66     8.718     8.775     8.832     8.888     8.944
         9     9.055      9.11     9.165      9.22     9.274     9.327     9.381
     9.434     9.487     9.539     9.592     9.644     9.695     9.747     9.798
     9.849     9.899      9.95        10     10.05      10.1     10.15      10.2
     10.25      10.3     10.34     10.39     10.44     10.49     10.54     10.58
     10.63     10.68     10.72     10.77     10.82     10.86     10.91     10.95
        11     11.05     11.09     11.14     11.18     11.22     11.27     11.31
     11.36      11.4     11.45     11.49     11.53     11.58     11.62     11.66
      11.7     11.75     11.79     11.83     11.87     11.92     11.96        12
     12.04     12.08     12.12     12.17     12.21     12.25     12.29     12.33
     12.37     12.41     12.45     12.49     12.53     12.57     12.61     12.65
     12.69     12.73     12.77     12.81     12.85     12.88     12.92     12.96
        13     13.04     13.08     13.11     13.15     13.19     13.23     13.27
      13.3     13.34     13.38     13.42     13.45     13.49     13.53     13.56
      13.6     13.64     13.67     13.71     13.75     13.78     13.82     13.86

*/
