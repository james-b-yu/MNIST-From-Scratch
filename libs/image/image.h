#include <inttypes.h>
#include <string>
#include <vector>

#include "../learn/learn.h"

#pragma once

struct Image : public Learn::TrainingExample {
  private:
	const std::string _shades = "$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\\|()1{}[]?-_+~<>i!lI;:,\"^`'. ";
	unsigned short _label;
	size_t _xSize;
	size_t _ySize;

  public:
	Image(std::vector<double> &&pixels, unsigned short label, size_t imageXSize, size_t imageYSize);
	Image(const std::vector<double> &pixels, unsigned short label, size_t imageXSize, size_t imageYSize);

	~Image();

	virtual std::string toString();
};