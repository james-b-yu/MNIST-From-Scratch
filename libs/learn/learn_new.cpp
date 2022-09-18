#include "learn.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <random>

Learn::TrainingExample::TrainingExample(const std::vector<double> &input, const std::vector<double> expectedOutput)
	: _input(input), _expectedOutput(expectedOutput)
{
}
Learn::TrainingExample::TrainingExample(std::vector<double> &&input, std::vector<double> &&expectedOutput)
	: _input(input), _expectedOutput(expectedOutput)
{
}

Learn::Result::Result(Vec &&outputLayer, double &&cost, std::vector<Mat> &&weightGradient,
					  std::vector<Vec> &&biasGradient, double accuracy)
	: _outputLayer(std::move(outputLayer)), _cost(std::move(cost)), _weightGradient(std::move(weightGradient)),
	  _biasGradient(std::move(biasGradient)), _accuracy(std::move(accuracy))
{
}

Learn::Learn(size_t numInputNodes, size_t numHiddenLayers, size_t numHiddenLayerNodes, size_t numOutputNodes,
			 std::vector<std::shared_ptr<TrainingExample>> &&trainingExamples, double learningRate)
	: _trainingExamples(std::move(trainingExamples)), _numInputNodes(numInputNodes), _numHiddenLayers(numHiddenLayers),
	  _numHiddenLayerNodes(numHiddenLayerNodes), _numOutputNodes(numOutputNodes), _numLayers(numHiddenLayers + 2),
	  _learningRate(learningRate)
{
	_weights = _initializeLikeWeights();
	_biases = _initializeLikeBiases();
}

std::vector<Learn::Mat> Learn::_initializeLikeWeights() const
{
	std::vector<Mat> weights;

	// keep the first layer empty since input layer should have no weights
	weights.push_back(Mat::Zero(0, 0));
	// for the first hidden layer
	weights.push_back(Mat::Zero(_numHiddenLayerNodes, _numInputNodes));
	// for the next layers
	for (size_t k = 1; k < _numHiddenLayers; ++k) {
		weights.push_back(Mat::Zero(_numHiddenLayerNodes, _numHiddenLayerNodes));
	}
	// for final layer
	weights.push_back(Mat::Zero(_numOutputNodes, _numHiddenLayerNodes));

	return weights;
}

std::vector<Learn::Vec> Learn::_initializeLikeLayers(const std::vector<double> &in) const
{
	// throw exception if in is wrong size
	if (in.size() && in.size() != _numInputNodes) {
		throw std::invalid_argument(
			"in vec must be same size as specifided, or empty"); // error if wrong vec size, accept empty vector
	}

	std::vector<Vec> layers;
	// first layer equals the input layer
	layers.push_back(Vec(in.size()));
	for (size_t i = 0; i < in.size(); ++i) {
		layers[0](i) = in[i];
	}

	// now add the hidden layers
	for (size_t k = 0; k < _numHiddenLayers; ++k) {
		layers.push_back(Vec::Zero(_numHiddenLayerNodes));
	}

	// now add output layer
	layers.push_back(Vec::Zero(_numOutputNodes));

	return layers;
}

std::vector<Learn::Vec> Learn::_initializeLikeBiases() const
{
	return _initializeLikeLayers({});
}

void Learn::_setRandomWeightsAndBiases()
{
	std::default_random_engine generator;
	std::uniform_real_distribution<double> dist(-0.1, 0.1);

	for (auto &k : _weights) {
		for (Eigen::Index i = 0; i < k.rows(); ++i) {
			for (Eigen::Index j = 0; j < k.cols(); ++j) {
				k(i, j) = dist(generator);
			}
		}
	}

	for (auto &k : _biases) {
		for (Eigen::Index i = 0; i < k.rows(); ++i) {
			k(i) = dist(generator);
		}
	}
}

Learn::Result Learn::_calculateIndividual(std::shared_ptr<TrainingExample> ex) const
{
	if (ex->_input.size() != _numInputNodes || ex->_expectedOutput.size() != _numOutputNodes) {
		throw std::invalid_argument("training example has wrong number of input or output nodes");
	}

	auto layers = _initializeLikeLayers(ex->_input);
	auto layersLinear = _initializeLikeLayers(ex->_input);
	for (size_t k = 1; k < _numLayers; ++k) {
		layersLinear[k] = _weights[k] * layers[k - 1] + _biases[k];
		layers[k] = layersLinear[k].unaryExpr(&Learn::_scalingFunc);
	}

	Vec outputLayer = layers.back();
	Vec outputLayerLinear = layersLinear.back();

	Vec expectedOutputLayer = Eigen::Map<Vec, Eigen::Unaligned>(ex->_expectedOutput.data(), _numOutputNodes);

	double cost = _costFunc(outputLayer, expectedOutputLayer);

	// backwards calculation
	std::vector<Vec> errorTerms = _initializeLikeLayers({});

	// final layer's error terms
	errorTerms[_numLayers - 1] = outputLayer.binaryExpr(expectedOutputLayer, &Learn::_dCostFunc)
									 .cwiseProduct(outputLayerLinear.unaryExpr(&Learn::_dScalingFunc));

	// previous layers' error terms
	for (size_t k = _numLayers - 2; k > 0; --k) {
		errorTerms[k] = (_weights[k + 1].transpose() * errorTerms[k + 1])
							.cwiseProduct(layersLinear[k].unaryExpr(&Learn::_dScalingFunc));
	}

	// get weight grad
	std::vector<Mat> weightGrad = _initializeLikeWeights();
	for (size_t k = 1; k < _numLayers; ++k) {
		weightGrad[k] = errorTerms[k] * layers[k - 1].transpose();
	}

	// bias grad is the same
	std::vector<Vec> biasGrad = std::move(errorTerms);

	// get accuracy, which for now is either 0 or 1
	size_t maxCalculatedIndex = 0;
	double maxCalculated = 0;
	size_t maxExpectedIndex = 0;
	double maxExpected = 0;
	for (Eigen::Index i = 0; i < outputLayer.size(); ++i) {
		if (outputLayer(i) > maxCalculated) {
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

	return Result(std::move(outputLayer), std::move(cost), std::move(weightGrad), std::move(errorTerms),
				  std::move(accuracy));
}

double Learn::_scalingFunc(const double &v)
{
	return 0.5 * (1.0 + std::tanh(v));
}

double Learn::_dScalingFunc(const double &v)
{
	return 0.5 / (std::pow(std::cosh(v), 2));
}

double Learn::_costFunc(const double &c, const double &e)
{
	return std::pow(c - e, 2);
}

double Learn::_dCostFunc(const double &c, const double &e)
{
	return 2.0 * (c - e);
}

double Learn::_costFunc(const Vec &c, const Vec &e)
{
	if (c.rows() != e.rows()) {
		throw std::invalid_argument("c and e have differing sizes");
	}

	double result = 0.0;

	for (Eigen::Index i = 0; i < c.rows(); ++i) {
		result += _costFunc(c(i), e(i));
	}

	return result;
}

Learn::Result Learn::_getAverageFromResults(const std::vector<Result> &res) const
{
	// get average cost and accuracy
	double cost = 0;
	double accuracy = 0;
	for (const auto &r : res) {
		cost += r._cost;
		accuracy += r._accuracy;
	}
	cost /= res.size();
	accuracy /= res.size();

	// get average weight and bias gradient
	std::vector<Mat> weightGradient = _initializeLikeWeights();
	std::vector<Vec> biasGradient = _initializeLikeBiases();

	for (const auto &r : res) {
		for (size_t k = 0; k < res.size(); ++k) {
			weightGradient[k] += r._weightGradient[k];
			biasGradient[k] += r._weightGradient[k];
		}
	}

	for (auto &i : weightGradient) {
		i /= res.size();
	}

	for (auto &i : biasGradient) {
		i /= res.size();
	}

	return Result({}, std::move(cost), std::move(weightGradient), std::move(biasGradient), std::move(accuracy));
}