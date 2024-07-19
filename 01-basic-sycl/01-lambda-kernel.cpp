#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>

// sycl::queue:
/*
	Submit a lambda to group commands (command group)
*/

// sycl::handler parallel_for:
/*
	Invoke a lambda kernel
*/

int main()
{
	sycl::queue queue{sycl::gpu_selector_v};

	queue.submit(
		[&] (sycl::handler & handler)
		{
			handler.parallel_for(
				sycl::range<1>{16},	// 1D data, 16 items
				[=] (sycl::item<1> item)	// a lambda kernel
				{
					sycl::id<1> id = item.get_id();
				}
			);
		}
	);
}

