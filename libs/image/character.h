#pragma once
#include <inttypes.h>
#include <string>
#include <vector>

#include "../learn/learn.h"

struct Character : public Learn<unsigned char>::TrainingExample {
  private:
	const std::string _shades = "$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\\|()1{}[]?-_+~<>i!lI;:,\"^`'. ";
	unsigned short _label;
	size_t _xSize;
	size_t _ySize;
	unsigned char _getCharFromLabel(unsigned short label) const;

  public:
	Character(const std::vector<double> &pixels, unsigned short label, size_t imageXSize, size_t imageYSize);
	virtual double calculateAbsoluteAccuracy(const std::vector<double> &calculatedOutput) const;
	virtual double calculateConfidence(const std::vector<double> &calculatedOutput) const;
	virtual unsigned char interpretOutput(const std::vector<double> &output) const;
	virtual std::string toString() const;
};
