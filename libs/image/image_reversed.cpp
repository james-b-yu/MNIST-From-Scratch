#include "image_reversed.h"
#include <cmath>
#include <numeric>
#include <sstream>

ImageReversed::ImageReversed(unsigned short label, const std::vector<double> &pixels, size_t imageXSize,
							 size_t imageYSize)
	: TrainingExample({}, pixels), _xSize(imageXSize), _ySize(imageYSize)
{
	_number = (2.0 * static_cast<double>(label) / 9) - 1;
	_input = {_number};
}

std::string ImageReversed::toString() const
{
	std::stringstream sstr;

	sstr << _number;

	return sstr.str();
}

std::string ImageReversed::interpretOutput(const std::vector<double> &output) const
{
	std::stringstream sstr;
	for (size_t y = 0; y < _ySize; ++y) {
		for (size_t x = 0; x < _xSize; ++x) {
			double pixelValue = output[y * _xSize + x];

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

double ImageReversed::calculateAbsoluteAccuracy(const std::vector<double> &calculatedOutput) const
{
	return 0.0;
}

double ImageReversed::calculateConfidence(const std::vector<double> &calculatedOutput) const
{
	return 0.0;
}