#include <inttypes.h>
#include <string>
#include <vector>

#include "../learn/learn.h"

#pragma once

struct ImageReversed : public Learn<std::string>::TrainingExample {
  private:
	const std::string _shades = "$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\\|()1{}[]?-_+~<>i!lI;:,\"^`'. ";
	double _number;
	size_t _xSize;
	size_t _ySize;

  public:
	ImageReversed(unsigned short label, const std::vector<double> &pixels, size_t imageXSize, size_t imageYSize);
	virtual double calculateAbsoluteAccuracy(const std::vector<double> &calculatedOutput) const;
	virtual double calculateConfidence(const std::vector<double> &calculatedOutput) const;
	virtual std::string interpretOutput(const std::vector<double> &output) const;
	virtual std::string toString() const;
};
