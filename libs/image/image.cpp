#include "image.h"
#include <cmath>
#include <sstream>

Image::Image(std::vector<double> &&pixels, unsigned short label, size_t imageXSize, size_t imageYSize)
	: TrainingExample(pixels, {}), _label(label), _xSize(imageXSize), _ySize(imageYSize)
{
	// set correct output node to 1.0; all others are 0 by default
	// _expectedOutput = std::vector<double>(10);
	// _expectedOutput[_label] = 1.0;
}

Image::Image(const std::vector<double> &pixels, unsigned short label, size_t imageXSize, size_t imageYSize)
	: TrainingExample(pixels, {}), _label(label), _xSize(imageXSize), _ySize(imageYSize)
{
	// set correct output node to 1.0; all others are 0 by default
	_expectedOutput = std::vector<double>(10);
	_expectedOutput[_label] = 1.0;
}

std::string Image::toString()
{
	std::stringstream sstr;
	for (size_t y = 0; y < _ySize; ++y) {
		for (size_t x = 0; x < _xSize; ++x) {
			double pixelValue = _input[y * _xSize + x];

			const size_t numShades = _shades.size();
			// convert from 0-255 to 0-69
			unsigned int shade = std::round(static_cast<double>(numShades - 1) * pixelValue);

			sstr << _shades[numShades - 1 - shade] << _shades[numShades - 1 - shade];
		}
		if (y + 1 != _ySize) {
			sstr << "\n";
		}
	}

	return sstr.str();
}

Image::~Image()
{
	// std::cout << "image got deleted" << std::endl;
}