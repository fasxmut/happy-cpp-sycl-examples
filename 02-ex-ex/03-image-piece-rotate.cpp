//
// Copyright (c) 2024 Fas Xmut (fasxmut at protonmail dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#include <sycl/sycl.hpp>
#include <SFML/Graphics.hpp>
#include <filesystem>
#include <iostream>
#include <boost/assert.hpp>
#include <vector>
#include <array>

using std::string_literals::operator""s;

// Piece Rotate
// c++ sycl
// ./prog 03-q3.jpg 03-q3-output.jpg

namespace gpu
{
constexpr auto area_size = 256u;
constexpr auto block_size = 16u;
constexpr auto lm_offset = 2u;

using color_type = std::array<unsigned char, 3>;

class image_type
{
private:
	std::vector<gpu::color_type> __image;
	unsigned int __width, __height;
public:
	image_type() = delete;
	image_type(const std::string & filename__)
	{
		sf::Image * image = new sf::Image;
		if (! image->loadFromFile(filename__))
		{
			delete image;
			throw std::runtime_error{"Can not load image: "s + filename__};
		}

		{
			auto [w, h] = image->getSize();
			__width = w;
			__height = h;
		}

		{
			for (unsigned int y=0; y<__height; ++y)
			{
				for (unsigned int x=0; x<__width; ++x)
				{
					const auto & color = image->getPixel(x, y);
					__image.push_back({color.r, color.g, color.b});
				}
			}
		}

		delete image;
	}
public:
	std::vector<gpu::color_type> & image()
	{
		return __image;
	}
	unsigned int width() const
	{
		return __width;
	}
	unsigned int height() const
	{
		return __height;
	}
};

class image_piece_rotate_kernel
{
private:
	sycl::accessor<gpu::color_type, 2, sycl::access_mode::read> __input;
	sycl::accessor<gpu::color_type, 2, sycl::access_mode::write> __output;
	sycl::local_accessor<gpu::color_type, 3> __lm;
public:
	image_piece_rotate_kernel(
		sycl::buffer<gpu::color_type, 2> & in_buffer__,
		sycl::buffer<gpu::color_type, 2> & out_buffer__,
		const sycl::range<3> & lm_range__,
		sycl::handler & handler__
	):
		__input{in_buffer__, handler__, sycl::read_only},
		__output{out_buffer__, handler__, sycl::write_only},
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

		auto y_start = static_cast<unsigned int>(gidy/gpu::area_size) * gpu::area_size;
		auto x_start = static_cast<unsigned int>(gidx/gpu::area_size) * gpu::area_size;

		auto src_gidy = gidx - x_start + y_start;
		auto src_gidx = gidy - y_start + x_start;
		
		color_type & lm0 = __lm[lidy][lidx][0];

		lm0 = __input[src_gidy][src_gidx];
		sycl::group_barrier(item.get_group(), sycl::memory_scope::work_group);

		__output[gidy][gidx] = lm0;
	}
};

}	// namespace gpu

int main(int argc, char * argv[])
try
{
	if (argc != 3)
		throw std::runtime_error{""s + argv[0] + " <input image> <output image>"};
	if (! std::filesystem::exists(argv[1]))
		throw std::runtime_error{"Input image does not exist: "s + argv[1]};

	gpu::image_type input_image{argv[1]};
	std::cout << "Input image size: " << input_image.width() << " x " << input_image.height() << std::endl;

	if (input_image.width() % gpu::area_size != 0 || input_image.height() % gpu::area_size != 0)
		throw std::runtime_error{"Input image size must be N * "s + std::to_string(gpu::area_size) + " , (N > 0, N is int)"};

	if (input_image.width() % gpu::block_size != 0 || input_image.height() % gpu::block_size != 0)
		throw std::runtime_error{"Input image size must be N * "s + std::to_string(gpu::block_size) + " , (N > 0, N is int)"};

	auto input_buffer = sycl::buffer<gpu::color_type, 2>{
		input_image.image().data(),
		sycl::range<2>{input_image.height(), input_image.width()}
	};

	auto output_buffer = sycl::buffer<gpu::color_type, 2>{
		sycl::range<2>{input_image.height(), input_image.width()}
	};

	sycl::queue queue{sycl::gpu_selector_v};

	queue.submit(
		[&] (sycl::handler & handler)
		{
			auto piece_rotate = gpu::image_piece_rotate_kernel{
				input_buffer,
				output_buffer,
				sycl::range<3>{gpu::block_size, gpu::block_size, gpu::lm_offset},
				handler
			};
			handler.parallel_for<class name1>(
				sycl::nd_range<2>{
					sycl::range<2>{input_image.height(), input_image.width()},
					sycl::range<2>{gpu::block_size, gpu::block_size}
				},
				piece_rotate
			);
		}
	);

	auto host_access = output_buffer.get_host_access();

	sf::Image output_image;
	output_image.create(input_image.width(), input_image.height());

	for (int j=0; j<input_image.height(); ++j)
	{
		for (int i=0; i<input_image.width(); ++i)
		{
			gpu::color_type color = host_access[j][i];
			output_image.setPixel(i, j, sf::Color{color[0], color[1], color[2]});
		}
	}

	if (! output_image.saveToFile(argv[2]))
		throw std::runtime_error{"Save output image to file "s + argv[2] + " error."};
}
catch (const std::exception & e)
{
	std::cerr << "--------------------------------------------------------------------------------\n";
	std::cerr << "std::exception:\n";
	std::cerr << e.what() << std::endl;
}

