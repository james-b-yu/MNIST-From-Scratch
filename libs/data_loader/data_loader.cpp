#include "data_loader.h"
#include <iostream>

DataLoader::DataLoader(const std::string &imagesPath, const std::string &labelsPath)
{
	std::cout << "data loader opening" << std::endl;
	_imagesFile.open(imagesPath, std::ios::in | std::ios::binary);
	_labelsFile.open(labelsPath, std::ios::in | std::ios::binary);

	_imagesBufferSize = _imagesFile.seekg(0, std::ios::end).tellg() - _imagesFile.seekg(0, std::ios::beg).tellg();
	_labelsBufferSize = _labelsFile.seekg(0, std::ios::end).tellg() - _labelsFile.seekg(0, std::ios::beg).tellg();

	std::cout << "i " << _imagesBufferSize << " l " << _labelsBufferSize << std::endl;

	// allocate memory for buffers
	_imagesBuffer = reinterpret_cast<char *>(malloc(_imagesBufferSize));
	_labelsBuffer = reinterpret_cast<char *>(malloc(_labelsBufferSize));

	// read files into memory
	_imagesFile.seekg(0, std::ios::beg).read((char *)_imagesBuffer, _imagesBufferSize);
	_labelsFile.seekg(0, std::ios::beg).read(reinterpret_cast<char *>(_labelsBuffer), _labelsBufferSize);

	// close files as they are no longer necessary

	// get metadata from files
	_numImages = _ReadFromMSBBuffer<uint32_t>(4, _imagesBuffer);
	// exception if num images != num labels
	auto numLabels = _ReadFromMSBBuffer<uint32_t>(4, _labelsBuffer);

	if (numLabels != _numImages) {
		throw std::invalid_argument("number of images != number of labels!");
	}

	_imageYSize = _ReadFromMSBBuffer<uint32_t>(8, _imagesBuffer);
	_imageXSize = _ReadFromMSBBuffer<uint32_t>(12, _imagesBuffer);
}

DataLoader::~DataLoader()
{
	std::cout << "data loader closing" << std::endl;

	_imagesFile.close();
	_labelsFile.close();
	free(_imagesBuffer);
	free(_labelsBuffer);
}

std::shared_ptr<Image> DataLoader::getImage(size_t nthImage) // beginning at n = 0
{
	// make a vector of pixels
	std::vector<double> pixels(_imageXSize * _imageYSize);
	for (size_t i = 0; i < _imageXSize * _imageYSize; ++i) {
		pixels[i] = static_cast<double>(
						_ReadFromMSBBuffer<uint8_t>(16 + nthImage * _imageXSize * _imageYSize + i, _imagesBuffer)) /
					255.0;
	}

	// get the label
	unsigned short label = _ReadFromMSBBuffer<uint8_t>(8 + nthImage, _labelsBuffer);

	return std::make_shared<Image>(
		pixels, label, _imageXSize,
		_imageYSize); // unfortunately, must return a copy due to necessity of changing edianness
}
