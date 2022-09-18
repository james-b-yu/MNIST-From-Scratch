// #define EIGEN_USE_MKL_ALL
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <memory>

#include "libs/data_loader/data_loader.h"
#include "libs/image/character.h"
#include "libs/image/linear.h"
#include "libs/learn/learn.h"

int main(int, char **)
{
	Eigen::initParallel();

	// make a bunch of linear stuff
	std::vector<std::shared_ptr<Learn<std::string>::TrainingExample>> trainingExamples;
	std::vector<std::shared_ptr<Learn<std::string>::TrainingExample>> testingExamples;

	// make test examples for linear regression of the form y = mx + c
	// model is fed m x and c, and must figure out the y

	for (double m = 0.01; m < 0.5; m += 0.007)
		for (double x = 0.01; x < 0.5; x += 0.007)
			for (double c = 0.01; c < 0.5; c += 0.007) {
				trainingExamples.push_back(std::make_shared<Linear>(m * x + c, m, x, c));
			}

	for (double m = 0.5; m < 1; m += 0.010)
		for (double x = 0.5; x < 1; x += 0.010)
			for (double c = 0.5; c < 1; c += 0.010) {
				testingExamples.push_back(std::make_shared<Linear>(m * x + c, m, x, c));
			}

	// for (int i = 0; i < 10; ++i)
	// 	trainingExamples.push_back(std::make_shared<Linear>(0.2, 0, 0, 0));
	// for (int i = 0; i < 10; ++i)
	// 	testingExamples.push_back(std::make_shared<Linear>(0.2, 0, 0, 0));

	Learn<std::string> l(3, 2, 20, 1, 0.01, 0.01, 0, std::move(trainingExamples), std::move(testingExamples), 2, 16,
						 "save-file-linear.txt");
	l.setRandomLayerConnections();
	l.train(100);
}
