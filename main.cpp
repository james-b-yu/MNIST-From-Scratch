// #define EIGEN_USE_MKL_ALL
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <memory>

#include "libs/data_loader/data_loader.h"
#include "libs/image/image.h"
#include "libs/learn/learn.h"

int main(int, char **)
{
	Eigen::initParallel();
	// extract training examples;
	std::vector<std::shared_ptr<Learn<double>::TrainingExample>> trainingExamples;
	std::vector<std::shared_ptr<Learn<double>::TrainingExample>> testingExamples;

	DataLoader trainLoader("./data/train-images.idx3-ubyte", "./data/train-labels.idx1-ubyte");
	DataLoader testLoader("./data/t10k-images.idx3-ubyte", "./data/t10k-labels.idx1-ubyte");

	for (size_t i = 0; i < 60000; ++i) {
		trainingExamples.push_back(trainLoader.getImage(i));
	}

	for (size_t i = 0; i < 10000; ++i) {
		testingExamples.push_back(testLoader.getImage(i));
	}

	Learn<double> l(28 * 28, 3, 512, 10, 0.01, 0.001, 0.8, std::move(trainingExamples), std::move(testingExamples), 2,
					16, "save-file.txt");

	try {
		l.loadLayerConnections();
	} catch (...) {
		l.setRandomLayerConnections();
	}

	l.train(12);
	for (size_t i = 0; i < 10; ++i)
		l.predictIndividual(testLoader.getImage(i));
}