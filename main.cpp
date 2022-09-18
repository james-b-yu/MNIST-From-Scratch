#include "libs/data_loader/data_loader.h"
#include "libs/image/image.h"
#include "libs/learn/learn.h"
#include "libs/linalg/linalg.h"

#include <iostream>
#include <memory>

int main(int, char **)
{
	// extract training examples;
	std::vector<std::shared_ptr<Learn::TrainingExample>> trainingExamples;

	DataLoader a("./data/train-images.idx3-ubyte", "./data/train-labels.idx1-ubyte");

	for (size_t i = 0; i < 60000; ++i) {
		trainingExamples.push_back(a.getImage(i));
	}

	Learn l(28 * 28, 2, 512, 10, std::move(trainingExamples), 0.01);
	l._setRandomWeightsAndBiases();

	l.train(50000);
	// std::cout << trainingExamples[50]->toString() << std::endl;

	// LinAlg::Matrix m({{1, 2, 3, 4}, {5, 6, 7, 8}});
	// LinAlg::Matrix n({{2, 3, 4, 5}, {6, 7, 8, 9}});
	// std::cout << LinAlg::hprod(m, n) << std::endl;
	// std::cout << LinAlg::mtrans(m) << std::endl << std::endl;

	// // dummy for tests
	// std::vector<std::shared_ptr<Learn::TrainingExample>> trainingExamples;
	// for (size_t i = 0; i < 1000; ++i) {
	// 	std::vector<double> in = {1, 2, 3, 4, 5};
	// 	std::vector<double> out = {0, 0, 1, 0};
	// 	trainingExamples.push_back(std::make_shared<Learn::TrainingExample>(in, out));
	// }

	// Learn l(5, 2, 4, 4, std::move(trainingExamples), 0.1);
	// l._setRandomWeightsAndBiases();
	// l.train(100);
}