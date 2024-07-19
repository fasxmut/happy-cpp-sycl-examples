#include <sycl/sycl.hpp>

class kernel_class
{
public:
	// A class object function operator can be sent to as kernel too.
	void operator()(sycl::item<1> item__) const
	{
		sycl::id<1> id = item__.get_id();
	}
};

int main()
{
	sycl::queue queue{sycl::gpu_selector_v};
	kernel_class kernel;
	queue.submit(
		[&] (sycl::handler & handler)
		{
			handler.parallel_for(sycl::range<1>{16}, kernel);
		}
	);
}

