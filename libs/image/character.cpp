#include "character.h"
#include <cmath>
#include <numeric>
#include <sstream>

// _label will range between 0 and 61 (so 62 output nodes)
Character::Character(const std::vector<double> &pixels, unsigned short label, size_t imageXSize, size_t imageYSize)
	: TrainingExample({}, {}), _label(label), _xSize(imageXSize), _ySize(imageYSize)
{

	_input = std::vector<double>(_xSize * _ySize);
	for (size_t x = 0; x < _xSize; ++x) {
		for (size_t y = 0; y < _ySize; ++y) {
			_input[y * _xSize + x] = pixels[x * _ySize + y];
		}
	}

	// set correct output node to 1.0; all others are 0 by default
	_expectedOutput = std::vector<double>(62);
	_expectedOutput[_label] = 1.0;
}

std::string Character::toString() const
{
	std::stringstream sstr;
	for (size_t y = 0; y < _ySize; ++y) {
		for (size_t x = 0; x < _xSize; ++x) {
			double pixelValue = 0.5 * (1 + _input[y * _xSize + x]);

			const size_t numShades = _shades.size();
			// convert from 0-255 to 0-69
			unsigned int shade = std::round(static_cast<double>(numShades - 1) * pixelValue);

			sstr << _shades[numShades - 1 - shade] << _shades[numShades - 1 - shade];
		}
		sstr << _getCharFromLabel(_label) << "\t" << (unsigned short)_label;
		if (y + 1 != _ySize) {
			sstr << "\n";
		}
	}

	return sstr.str();
}

unsigned char Character::interpretOutput(const std::vector<double> &output) const
{
	unsigned short index = static_cast<char>(std::max_element(output.begin(), output.end()) - output.begin());
	return _getCharFromLabel(index);
}

double Character::calculateAbsoluteAccuracy(const std::vector<double> &calculatedOutput) const
{
	size_t maxCalculatedIndex = interpretOutput(calculatedOutput);
	size_t maxExpectedIndex = interpretOutput(_expectedOutput);

	double accuracy = (maxExpectedIndex == maxCalculatedIndex) ? 1.0 : 0.0;
	return accuracy;
}

double Character::calculateConfidence(const std::vector<double> &calculatedOutput) const
{
	double maxCalculated = 0;
	for (size_t i = 0; i < calculatedOutput.size(); ++i) {
		if (calculatedOutput[i] > maxCalculated) {
			maxCalculated = calculatedOutput[i];
		}
	}

	double totalCalculated = std::accumulate(calculatedOutput.begin(), calculatedOutput.end(), 0.0);

	double confidence = maxCalculated / totalCalculated;
	return confidence;
}

unsigned char Character::_getCharFromLabel(unsigned short label) const
{
	if (label < 10) {
		return label + 48;
	} else if (label < 37) {
		return label + 55;
	} else {
		return label + 69;
	}
}