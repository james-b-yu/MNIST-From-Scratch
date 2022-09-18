#include "learn.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <random>

Learn::Learn(size_t numInputNodes, size_t numHiddenLayers, size_t numHiddenLayerNodes, size_t numOutputNodes,
			 std::vector<std::shared_ptr<TrainingExample>> &&trainingExamples, double learningRate)
	: _numInputNodes(numInputNodes), _numHiddenLayers(numHiddenLayers), _numHiddenLayerNodes(numHiddenLayerNodes),
	  _numOutputNodes(numOutputNodes), _numLayers(numHiddenLayers + 2), _trainingExamples(std::move(trainingExamples)),
	  _learningRate(learningRate)
{
	_initializeLikeWeights(_weights);
	_initializeLikeBiases(_biases);
}

void Learn::_initializeLikeWeights(std::vector<LinAlg::Matrix> &weights) const
{
	// set up weights.
	// the 0th layer is the input layer, so add a dummy empty matrix
	// for the ith layer, a matrix of size m = r^i, n = r^(i - 1) is created
	// first dimension represents nodes in the ith layer; second dimension represents nodes in the (i - 1)th layer
	weights.push_back(LinAlg::Matrix(0, 0)); // empty matrix

	weights.push_back(LinAlg::Matrix(_numHiddenLayerNodes, _numInputNodes));

	for (size_t i = 1; i < _numHiddenLayers; ++i) {
		weights.push_back(LinAlg::Matrix(_numHiddenLayerNodes, _numHiddenLayerNodes));
	}
	weights.push_back(LinAlg::Matrix(_numOutputNodes, _numHiddenLayerNodes)); // output layer
}

void Learn::_initializeLikeBiases(std::vector<LinAlg::Vector> &biases) const
{
	// set up biases.
	biases.push_back({}); // add dummy empty bias for input node
	for (size_t i = 1; i <= _numHiddenLayers; ++i) {
		biases.push_back(LinAlg::Vector(_numHiddenLayerNodes));
	}
	biases.push_back(LinAlg::Vector(_numOutputNodes));
}

void Learn::_initializeLikeLayers(std::vector<LinAlg::Vector> &layers, const LinAlg::Vector &in) const
{
	// create layers
	layers.push_back(LinAlg::Vector(_numInputNodes)); // input nodes
	for (size_t i = 1; i <= _numHiddenLayers; ++i) {
		layers.push_back(LinAlg::Vector(_numHiddenLayerNodes)); // hidden layers
	}
	layers.push_back(LinAlg::Vector(_numOutputNodes)); // output layers

	// set the input layer
	if (in.size() && in.size() != _numInputNodes)
		throw std::invalid_argument(
			"in vec must be same size as specifided, or empty"); // error if wrong vec size, but accept empty vector for
																 // the backpropagation stage

	layers[0] = in;
}

void Learn::_setRandomWeightsAndBiases()
{
	std::default_random_engine generator;
	std::uniform_real_distribution<double> dist(-0.1, 0.1);

	for (LinAlg::Matrix &i : _weights) {
		for (size_t j = 0; j < i.size().first; ++j) {
			for (size_t k = 0; k < i.size().second; ++k) {
				i[j][k] = dist(generator);
			}
		}
	}

	for (auto &i : _biases) {
		for (size_t j = 0; j < i.size(); ++j) {
			i[j] = 0;
		}
	}
}

double Learn::_scalingFunc(const double &v) const
{
	return 0.5 * (1.0 + std::tanh(v));
	// return v;
	// return std::max(0.0, v);

	// return 1.0 / (1.0 + std::exp(-v));
}

LinAlg::Vector Learn::_scalingFunc(const LinAlg::Vector &v) const
{
	LinAlg::Vector result(v.size());
	for (size_t i = 0; i < v.size(); ++i) {
		result[i] = _scalingFunc(v[i]);
	}

	return result;
}

double Learn::_scalingFuncOutput(const double &v) const
{
	return 0.5 * (1.0 + std::tanh(v));
	// return v;
	// return std::max(0.0, v);

	// return 1.0 / (1.0 + std::exp(-v));
}

LinAlg::Vector Learn::_scalingFuncOutput(const LinAlg::Vector &v) const
{
	LinAlg::Vector result(v.size());
	for (size_t i = 0; i < v.size(); ++i) {
		result[i] = _scalingFuncOutput(v[i]);
	}

	return result;
}

double Learn::_dScalingFunc(const double &v) const
{
	return 0.5 / std::pow(std::cosh(v), 2);
	// return 1;

	// if (v > 0)
	// 	return 1;
	// else
	// 	return 0;

	// return _scalingFunc(v) * (1.0 - _scalingFunc(v));
}

double Learn::_dScalingFuncOutput(const double &v) const
{
	return 0.5 / std::pow(std::cosh(v), 2);
	// return 1;

	// if (v > 0)
	// 	return 1;
	// else
	// 	return 0;

	// return _scalingFunc(v) * (1.0 - _scalingFunc(v));
}

LinAlg::Vector Learn::_dScalingFunc(const LinAlg::Vector &v) const
{
	LinAlg::Vector result(v.size());
	for (size_t i = 0; i < v.size(); ++i) {
		result[i] = _dScalingFunc(v[i]);
	}

	return result;
}

LinAlg::Vector Learn::_dScalingFuncOutput(const LinAlg::Vector &v) const
{
	LinAlg::Vector result(v.size());
	for (size_t i = 0; i < v.size(); ++i) {
		result[i] = _dScalingFuncOutput(v[i]);
	}

	return result;
}

double Learn::_costFunc(const double &calculated, const double &expected) const
{
	return std::pow(calculated - expected, 2.0);
}

double Learn::_dCostFunc(const double &calculated, const double &expected) const
{
	return 2.0 * (calculated - expected);
}

double Learn::_costFunc(const LinAlg::Vector &calculated, const LinAlg::Vector &expected) const
{
	if (calculated.size() != expected.size()) {
		throw std::invalid_argument("calculated and expected have different sizes");
	}

	double sum = 0.0;

	for (size_t i = 0; i < calculated.size(); ++i) {
		sum += _costFunc(calculated[i], expected[i]);
	}

	return sum;
}

LinAlg::Vector Learn::_dCostFunc(const LinAlg::Vector &calculated, const LinAlg::Vector &expected) const
{
	if (calculated.size() != expected.size()) {
		throw std::invalid_argument("calculated and expected have different sizes");
	}

	LinAlg::Vector result(calculated.size());
	for (size_t i = 0; i < calculated.size(); ++i) {
		result[i] = _dCostFunc(calculated[i], expected[i]);
	}

	return result;
}

Learn::Result Learn::_calculateIndividual(std::shared_ptr<TrainingExample> ex) const
{
	// ======== forward calculation
	// size of in should be correct
	if (ex->_input.size() != _numInputNodes) {
		throw std::invalid_argument("input has wrong number of nodes");
	}

	if (ex->_expectedOutput.size() != _numOutputNodes) {
		throw std::invalid_argument("output has wrong number of nodes");
	}

	// create layers
	std::vector<LinAlg::Vector> layers;
	_initializeLikeLayers(layers, ex->_input);

	// untransformed layers
	std::vector<LinAlg::Vector> layersLinear;
	_initializeLikeLayers(layersLinear, ex->_input);

	// forward calculation
	for (size_t k = 1; k < _numLayers; ++k) {
		layersLinear[k] = (LinAlg::Vector)LinAlg::mmul(_weights[k], layers[k - 1]) + _biases[k];
		if (k + 1 == _numLayers)
			layers[k] = _scalingFuncOutput(layersLinear[k]);
		else
			layers[k] = _scalingFunc(layersLinear[k]);
	}

	// get final layer
	LinAlg::Vector outputLayer = layers.back();
	LinAlg::Vector outputLayerLinear = layersLinear.back();

	// calculate cost
	double cost = _costFunc(outputLayer, ex->_expectedOutput);

	// backwards calculation

	std::vector<LinAlg::Vector> errorTerms;
	_initializeLikeLayers(errorTerms, {});

	// final layer
	errorTerms[_numLayers - 1] =
		LinAlg::hprod(_dCostFunc(outputLayer, ex->_expectedOutput), _dScalingFuncOutput(outputLayerLinear));

	// intermediate layers
	for (size_t k = _numLayers - 2; k > 0; --k) {
		errorTerms[k] = LinAlg::hprod((LinAlg::Vector)LinAlg::mmul(LinAlg::mtrans(_weights[k + 1]), errorTerms[k + 1]),
									  _dScalingFunc(layersLinear[k]));
	}

	// weights
	std::vector<LinAlg::Matrix> weightGradients;
	_initializeLikeWeights(weightGradients);
	for (size_t k = 1; k < _numLayers; ++k) {
		for (size_t j = 0; j < layers[k].size(); ++j) {
			for (size_t i = 0; i < layers[k - 1].size(); ++i) {
				weightGradients[k][j][i] = layers[k - 1][i] * errorTerms[k][j];
			}
		}
	}

	size_t maxCalculatedIndex = 0;
	double maxCalculated = 0;
	size_t maxExpectedIndex = 0;
	double maxExpected = 0;
	for (size_t i = 0; i < outputLayer.size(); ++i) {
		if (outputLayer[i] > maxCalculated) {
			maxCalculated = outputLayer[i];
			maxCalculatedIndex = i;
		}
	}

	for (size_t i = 0; i < ex->_expectedOutput.size(); ++i) {
		if (ex->_expectedOutput[i] > maxExpected) {
			maxExpected = ex->_expectedOutput[i];
			maxExpectedIndex = i;
		}
	}

	double accuracy = (maxExpectedIndex == maxCalculatedIndex) ? 1.0 : 0.0;

	return Result(std::move(outputLayer), std::move(cost), std::move(weightGradients), std::move(errorTerms),
				  std::move(accuracy));
}

Learn::Result::Result(LinAlg::Vector &&outputLayer, double &&cost, std::vector<LinAlg::Matrix> &&weightGradient,
					  std::vector<LinAlg::Vector> &&biasGradient, double accuracy)
	: _outputLayer(std::move(outputLayer)), _cost(std::move(cost)), _weightGradient(std::move(weightGradient)),
	  _biasGradient(std::move(biasGradient)), _accuracy(accuracy)
{
}

Learn::TrainingExample::TrainingExample(const std::vector<double> &input, const std::vector<double> expectedOutput)
	: _input(input), _expectedOutput(expectedOutput)
{
}
Learn::TrainingExample::TrainingExample(std::vector<double> &&input, std::vector<double> &&expectedOutput)
	: _input(input), _expectedOutput(expectedOutput)
{
}

std::vector<Learn::Result> Learn::_calculateMultiple(std::vector<std::shared_ptr<TrainingExample>> exs) const
{
	const size_t numExamples = exs.size();
	std::vector<Result> res;
	res.reserve(numExamples);

	for (const auto &ex : exs) {
		res.push_back(_calculateIndividual(ex));
	}

	return res;
}

void Learn::train(size_t epochs)
{

	// for (size_t epoch = 0; epoch < epochs; ++epoch) {
	// }

	// std::vector<std::shared_ptr<TrainingExample>> shortlist(&_trainingExamples[0], &_trainingExamples[2]);
	// auto res = _calculateMultiple(shortlist);

	// for (auto i : res[0]._weightGradient)
	// 	std::cout << i << std::endl;

	// create the threads
	std::vector<std::thread> threads;

	for (size_t t = 0; t < _concurrency; ++t) {
		threads.push_back(std::thread(&Learn::_threadWorkerFunc, this, epochs));
	}

	for (auto &t : threads) {
		t.join();
	}
}

// void Learn::train(size_t epochs)
// {
// 	for (size_t epochNumber = 1; epochNumber <= epochs; ++epochNumber) {
// 		// shuffle
// 		std::vector<std::shared_ptr<TrainingExample>> queue(_trainingExamples.begin(), _trainingExamples.end());
// 		std::shuffle(queue.begin(), queue.end(), std::default_random_engine(0));

// 		const size_t maxBatchSize = 128;

// 		while (!queue.empty()) {
// 			size_t batchSize = std::min(maxBatchSize, queue.size());
// 			std::vector<std::shared_ptr<TrainingExample>> batch =
// 				std::vector<std::shared_ptr<TrainingExample>>(queue.begin(), queue.begin() + batchSize);
// 			queue.erase(queue.begin(), queue.begin() + batchSize);

// 			auto results = _calculateMultiple(std::move(batch));
// 			auto average = _getAverageFromResults(results);

// 			// apply

// 			std::cout << "epoch " << epochNumber << " cost " << average._cost << " accuracy " << average._accuracy
// 					  << std::endl;

// 			for (size_t i = 0; i < _biases.size(); ++i) {
// 				_biases[i] = _biases[i] + (-1) * average._biasGradient[i] * _learningRate;
// 			}

// 			for (size_t i = 0; i < _weights.size(); ++i) {
// 				_weights[i] = _weights[i] + (-1) * average._weightGradient[i] * _learningRate;
// 			}
// 		}
// 	}
// }

std::vector<std::shared_ptr<Learn::TrainingExample>> Learn::_getNextExamples()
{
	// const size_t maxExampleGroupSize = std::ceil(0.01 * static_cast<double>(_trainingExamples.size()) /
	// static_cast<double>(_concurrency));
	const size_t maxExampleGroupSize = 4;

	std::unique_lock<std::mutex> queueLock(_exampleQueueMutex);
	size_t numRemainingExamples = _exampleQueue.size();

	// first check whether count has reached
	if ((_threadsTaken == _concurrency || numRemainingExamples == 0) && _threadsTaken != 0) {
		// wait for all the threads to have added to the results vector
		std::unique_lock<std::mutex> resultLock(_threadResultsMutex);
		_threadResultsConditionVariable.wait(resultLock, [&]() { // cv will get notified every time a result gets pushed
			// std::cout << "should not get callled alll the itme" << std::endl;
			return this->_threadResults.size() == _threadsTaken; // so wait until all results have arrived
		});

		// all results have now arrived, so find average gradient and learn

		Result average = _getAverageFromResults(_threadResults);
		std::cout << "epoch " << _epochNumber << " cost " << average._cost << " accuracy " << average._accuracy
				  << " remaining " << numRemainingExamples << std::endl;

		// apply
		for (size_t i = 0; i < _biases.size(); ++i) {
			_biases[i] = _biases[i] + (-1) * average._biasGradient[i] * _learningRate;
		}

		for (size_t i = 0; i < _weights.size(); ++i) {
			_weights[i] = _weights[i] + (-1) * average._weightGradient[i] * _learningRate;
		}

		_threadsTaken = 0;
		_threadResults.clear();
	}

	// not all threads have been taken up
	_threadsTaken++;
	if (numRemainingExamples == 0) { // if new epoch, add all to queue and randomize
		_epochNumber++;
		_exampleQueue =
			std::deque<std::shared_ptr<TrainingExample>>(_trainingExamples.begin(), _trainingExamples.end());
		// shuffle
		std::shuffle(_exampleQueue.begin(), _exampleQueue.end(), std::default_random_engine(0));
		numRemainingExamples = this->_trainingExamples.size();
	}

	// create list of next examples
	size_t exampleGroupSize = std::min(numRemainingExamples, maxExampleGroupSize);
	std::vector<std::shared_ptr<TrainingExample>> nextExamples(_exampleQueue.begin(),
															   _exampleQueue.begin() + exampleGroupSize);
	_exampleQueue.erase(_exampleQueue.begin(), _exampleQueue.begin() + exampleGroupSize);

	return nextExamples;
}

void Learn::_threadWorkerFunc(size_t epochs)
{
	while (_epochNumber <= epochs) { // todo: make this variable
		auto nextExamples = _getNextExamples();
		auto results = _calculateMultiple(nextExamples);

		// find average of results
		Result average = _getAverageFromResults(results);

		std::unique_lock<std::mutex> resultsLock(_threadResultsMutex);
		_threadResults.push_back(std::move(average));
		_threadResultsConditionVariable.notify_one();
	}
}

Learn::Result Learn::_getAverageFromResults(const std::vector<Result> &results) const
{
	Result average = results[0];
	if (results.size() == 1) {
		return average;
	}

	for (size_t r = 1; r < results.size(); ++r) {
		average._cost += results[r]._cost;
		average._accuracy += results[r]._accuracy;

		for (size_t i = 0; i < average._biasGradient.size(); ++i) {
			average._biasGradient[i] = average._biasGradient[i] + results[r]._biasGradient[i];
		}

		for (size_t i = 0; i < average._weightGradient.size(); ++i) {
			average._weightGradient[i] = average._weightGradient[i] + results[r]._weightGradient[i];
		}
	}

	average._cost /= results.size();
	average._accuracy /= results.size();

	for (auto &i : average._biasGradient) {
		i = i / results.size();
	}

	for (auto &i : average._weightGradient) {
		i = i / results.size();
	}

	return average;
}