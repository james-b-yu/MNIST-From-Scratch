#include <boost/endian/conversion.hpp>
#include <fstream>
#include <memory>
#include <string>

#include "../image/image.h"

class DataLoader
{
  private:
	std::ifstream _imagesFile;
	std::ifstream _labelsFile;
	// std::unique_ptr<void> _imagesBuffer;
	// std::unique_ptr<void> _labelsBuffer;
	char *_imagesBuffer;
	char *_labelsBuffer;

	size_t _imagesBufferSize;
	size_t _labelsBufferSize;

	uint32_t _imageXSize;
	uint32_t _imageYSize;

	template <typename T> T _ReadFromMSBBuffer(size_t bufferPos, char *buffer)
	{
		return boost::endian::big_to_native(*reinterpret_cast<T *>(buffer + bufferPos));
	}

  public:
	DataLoader(const std::string &imagesPath, const std::string &labelsPath);
	~DataLoader();

	uint32_t _numImages;
	std::shared_ptr<Image> getImage(size_t nthImage);
};