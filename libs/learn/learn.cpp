#include "learn.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <iostream>
#include <random>

template <typename T>
Learn<T>::TrainingExample::TrainingExample(const std::vector<double> &input, const std::vector<double> expectedOutput)
	: _input(input), _expectedOutput(expectedOutput)
{
}

template <typename T>
Learn<T>::TrainingExample::TrainingExample(std::vector<double> &&input, std::vector<double> &&expectedOutput)
	: _input(input), _expectedOutput(expectedOutput)
{
}

template <typename T>
Learn<T>::CalculationResult::CalculationResult(Vec &&outputLayer, double &&cost, std::vector<Mat> &&weightGradient,
											   std::vector<Vec> &&biasGradient, double &&accuracy,
											   double &&singleCalculationSpeed)
	: _outputLayer(std::move(outputLayer)), _cost(std::move(cost)), _weightGradient(std::move(weightGradient)),
	  _biasGradient(std::move(biasGradient)), _accuracy(std::move(accuracy)),
	  _singleCalculationSpeed(std::move(singleCalculationSpeed))
{
}

template <typename T>
Learn<T>::CalculationResult::CalculationResult(const Vec &outputLayer, const double &cost,
											   const std::vector<Mat> &weightGradient,
											   const std::vector<Vec> &biasGradient, const double &accuracy,
											   const double &singleCalculationSpeed)
	: _outputLayer(outputLayer), _cost(cost), _weightGradient(weightGradient), _biasGradient(biasGradient),
	  _accuracy(accuracy), _singleCalculationSpeed(singleCalculationSpeed)
{
}

template <typename T> Learn<T>::CalculationResult::CalculationResult(CalculationResult &&r)
{
	_outputLayer = std::move(r._outputLayer);
	_cost = std::move(r._cost);
	_weightGradient = std::move(r._weightGradient);
	_biasGradient = std::move(r._biasGradient);
	_accuracy = std::move(r._accuracy);
	_singleCalculationSpeed = std::move(r._singleCalculationSpeed);
}

template <typename T> void Learn<T>::CalculationResult::operator=(CalculationResult &&r)
{
	_outputLayer = std::move(r._outputLayer);
	_cost = std::move(r._cost);
	_weightGradient = std::move(r._weightGradient);
	_biasGradient = std::move(r._biasGradient);
	_accuracy = std::move(r._accuracy);
	_singleCalculationSpeed = std::move(r._singleCalculationSpeed);
}

template <typename T> void Learn<T>::CalculationResult::operator=(const CalculationResult &r)
{
	_outputLayer = r._outputLayer;
	_cost = r._cost;
	_weightGradient = r._weightGradient;
	_biasGradient = r._biasGradient;
	_accuracy = r._accuracy;
	_singleCalculationSpeed = r._singleCalculationSpeed;
}

template <typename T> typename Learn<T>::CalculationResult Learn<T>::EmptyResult()
{
	Vec outputLayer(_numOutputNodes);

	std::vector<Vec> biasGradients = _initializeLikeBiases();
	std::vector<Mat> weightGradients = _initializeLikeWeights();

	double cost = 0.0;
	double accuracy = 0.0;
	double singleCalculationSpeed = 0.0;

	return CalculationResult(std::move(outputLayer), std::move(cost), std::move(weightGradients),
							 std::move(biasGradients), std::move(accuracy), std::move(singleCalculationSpeed));
}

template <typename T>
Learn<T>::Learn(size_t numInputNodes, size_t numHiddenLayers, size_t numHiddenLayerNodes, size_t numOutputNodes,
				double learningRate, double decay, double momentum,
				std::vector<std::shared_ptr<TrainingExample>> &&trainingExamples,
				std::vector<std::shared_ptr<TrainingExample>> &&testingExamples, size_t concurrency,
				size_t batchSizePerThread, std::string saveFileLocation)
	: _numInputNodes(numInputNodes), _numHiddenLayers(numHiddenLayers), _numHiddenLayerNodes(numHiddenLayerNodes),
	  _numOutputNodes(numOutputNodes), _numLayers(numHiddenLayers + 2), _decay(decay), _momentum(momentum),
	  _trainingExamples(std::move(trainingExamples)), _testingExamples(testingExamples), _concurrency(concurrency),
	  _batchSizePerThread(batchSizePerThread), _saveFileLocation(saveFileLocation)
{
	_trainingState._weights = _initializeLikeWeights();
	_trainingState._biases = _initializeLikeBiases();
	_trainingState._learningRate = learningRate;

	// initialize multithreading data pool

	for (size_t i = 0; i < _concurrency + 1; ++i) {
		std::vector<CalculationResult> individualResultStorage;
		for (size_t j = 0; j < _batchSizePerThread; ++j) {
			individualResultStorage.push_back(std::move(EmptyResult()));
		}
		_threadIndividualResultStorage.push_back(std::move(individualResultStorage));

		_threadResults.push_back(std::move(EmptyResult()));

		// create blank thread data
		ThreadData td;
		std::vector<double> blankInputLayer(_numInputNodes);

		td._layers = _initializeLikeLayers(blankInputLayer);
		td._layersLinear = _initializeLikeLayers(blankInputLayer);
		td._biases = _initializeLikeBiases();
		td._weights = _initializeLikeWeights();
		td._errorTerms = _initializeLikeLayers(blankInputLayer);
		td._weightGrad = _initializeLikeWeights();
		td._biasGrad = _initializeLikeBiases();

		_threadWorkerData.push_back(std::move(td));
	}
}

template <typename T> std::vector<Mat> Learn<T>::_initializeLikeWeights() const
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

template <typename T> std::vector<Vec> Learn<T>::_initializeLikeLayers(const std::vector<double> &in) const
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

template <typename T> std::vector<Vec> Learn<T>::_initializeLikeBiases() const
{
	return _initializeLikeLayers({});
}

template <typename T> void Learn<T>::setRandomLayerConnections()
{
	std::default_random_engine generator;

	_trainingState._epochNumber = 0;

	for (auto &k : _trainingState._weights) {
		for (Eigen::Index i = 0; i < k.rows(); ++i) {
			for (Eigen::Index j = 0; j < k.cols(); ++j) {
				double r = std::sqrt(6.0 / (k.cols() + k.rows()));
				std::uniform_real_distribution<double> dist(-0.5 * r, 0.5 * r);
				k(i, j) = dist(generator);
			}
		}
	}

	for (auto &k : _trainingState._biases) {
		for (Eigen::Index i = 0; i < k.rows(); ++i) {
			k(i) = 0;
		}
	}
}

template <typename T>
typename Learn<T>::CalculationResult Learn<T>::_calculateIndividual(std::shared_ptr<TrainingExample> ex) const
{
	auto now = std::chrono::system_clock::now();
	// std::this_thread::sleep_for(std::chrono::milliseconds(10));

	// std::default_random_engine generator;
	// std::uniform_real_distribution<double> dist(-0.1, 0.1);
	// int a = 0;
	// std::vector<Learn::Mat> weights;

	// for (size_t i = 0; i < 1000; ++i) {
	// 	weights = _initializeLikeWeights();
	// 	a += dist(generator);
	// }

	if (ex->_input.size() != _numInputNodes || ex->_expectedOutput.size() != _numOutputNodes) {
		throw std::invalid_argument("training example has wrong number of input or output nodes");
	}

	Vec inputLayer = Eigen::Map<Vec, Eigen::Unaligned>(ex->_input.data(), _numInputNodes);

	auto &threadWorkerData = _threadWorkerData[_getThreadIndex()];

	threadWorkerData._layers[0] = inputLayer;
	threadWorkerData._layersLinear[0] = inputLayer;

	for (size_t k = 1; k < _numLayers; ++k) {
		threadWorkerData._layersLinear[k] =
			_trainingState._weights[k] * threadWorkerData._layers[k - 1] + _trainingState._biases[k];
		if (k + 1 != _numLayers)
			threadWorkerData._layers[k] = threadWorkerData._layersLinear[k].unaryExpr(&Learn::_scalingFunc);
		else
			threadWorkerData._layers[k] = threadWorkerData._layersLinear[k].unaryExpr(&Learn::_scalingFuncOutput);
	}

	Vec &outputLayer = threadWorkerData._layers.back();
	Vec &outputLayerLinear = threadWorkerData._layersLinear.back();

	Vec expectedOutputLayer = Eigen::Map<Vec, Eigen::Unaligned>(ex->_expectedOutput.data(), _numOutputNodes);
	std::vector<double> stdOutputLayer =
		std::vector<double>(outputLayer.data(), outputLayer.data() + outputLayer.size());

	double cost = outputLayer.binaryExpr(expectedOutputLayer, &Learn::_costFunc).sum();

	// backwards calculation
	// final layer's error terms
	threadWorkerData._errorTerms[_numLayers - 1] =
		outputLayer.binaryExpr(expectedOutputLayer, &Learn::_dCostFunc)
			.cwiseProduct(outputLayerLinear.unaryExpr(&Learn::_dScalingFuncOutput));

	// previous layers' error terms
	for (size_t k = _numLayers - 2; k > 0; --k) {
		threadWorkerData._errorTerms[k] =
			(_trainingState._weights[k + 1].transpose() * threadWorkerData._errorTerms[k + 1])
				.cwiseProduct(threadWorkerData._layersLinear[k].unaryExpr(&Learn::_dScalingFunc));
	}

	// get weight grad
	for (size_t k = 1; k < _numLayers; ++k) {
		threadWorkerData._weightGrad[k] = threadWorkerData._errorTerms[k] * threadWorkerData._layers[k - 1].transpose();
	}

	// bias grad is the same
	std::vector<Vec> &biasGrad = threadWorkerData._errorTerms;

	// get accuracy, which for now is either 0 or 1
	double accuracy = ex->calculateAbsoluteAccuracy(stdOutputLayer);
	auto singleCalculationSpeed = 1.0 / std::chrono::duration<double>(std::chrono::system_clock::now() - now).count();

	return CalculationResult(outputLayer, cost, threadWorkerData._weightGrad, biasGrad, accuracy,
							 singleCalculationSpeed);
}

template <typename T> double Learn<T>::_scalingFunc(const double &v)
{
	return v;
	return std::max(v, 0.0); // use relu
	return tanh(v);			 // use sigmoid between -1 and 1
}

template <typename T> double Learn<T>::_dScalingFunc(const double &v)
{
	return 1;
	if (v > 0) // use relu
		return 1;
	else
		return 0;
	return 1 / (std::pow(std::cosh(v), 2)); // use sigmoid between -1 and 1
}

template <typename T> double Learn<T>::_scalingFuncOutput(const double &v)
{
	return v;
	return 0.5 * (1.0 + std::tanh(v)); // sigmoid between 0 and 1
}

template <typename T> double Learn<T>::_dScalingFuncOutput(const double &v)
{
	return 1;
	return 0.5 / (std::pow(std::cosh(v), 2)); // sigmoid between 0 and 1
}

template <typename T> double Learn<T>::_costFunc(const double &c, const double &e)
{
	return std::pow(c - e, 2);
}

template <typename T> double Learn<T>::_dCostFunc(const double &c, const double &e)
{
	return 2.0 * (c - e);
}

// double Learn::_costFunc(const Vec &c, const Vec &e)
// {
// 	if (c.rows() != e.rows()) {
// 		throw std::invalid_argument("c and e have differing sizes");
// 	}

// 	double result = 0.0;

// 	for (Eigen::Index i = 0; i < c.rows(); ++i) {
// 		result += _costFunc(c(i), e(i)); // todo: could use eigen binary expr then sum isntead
// 	}

// 	return result;
// }

template <typename T>
typename Learn<T>::CalculationResult Learn<T>::_getAverageFromResults(
	const typename std::vector<CalculationResult>::iterator &begin,
	const typename std::vector<CalculationResult>::iterator &end) const
{
	const size_t resSize = end - begin;

	// get average cost and accuracy
	double cost = 0.0;
	double accuracy = 0.0;
	double singleCalculationSpeed = 0.0;
	for (auto it = begin; it != end; ++it) {
		cost += it->_cost;
		accuracy += it->_accuracy;
		singleCalculationSpeed += it->_singleCalculationSpeed;
	}
	cost /= resSize;
	accuracy /= resSize;
	singleCalculationSpeed /= resSize;

	// get average weight and bias gradient
	std::vector<Mat> weightGradient = _initializeLikeWeights();
	std::vector<Vec> biasGradient = _initializeLikeBiases();

	for (auto it = begin; it != end; ++it) {
		for (size_t k = 1; k < it->_weightGradient.size(); ++k) {
			weightGradient[k] += it->_weightGradient[k];
		}
		for (size_t k = 1; k < it->_biasGradient.size(); ++k) {
			biasGradient[k] += it->_biasGradient[k];
		}
	}

	for (auto &i : weightGradient) {
		i /= resSize;
	}

	for (auto &i : biasGradient) {
		i /= resSize;
	}

	return CalculationResult({}, std::move(cost), std::move(weightGradient), std::move(biasGradient),
							 std::move(accuracy), std::move(singleCalculationSpeed));
}

template <typename T>
typename Learn<T>::CalculationResult Learn<T>::_calculateMultiple(
	std::vector<std::shared_ptr<TrainingExample>> exs) const
{

	for (size_t i = 0; i < exs.size(); ++i) {
		_threadIndividualResultStorage[_getThreadIndex()][i] = _calculateIndividual(exs[i]);
	}

	return _getAverageFromResults(_threadIndividualResultStorage[_getThreadIndex()].begin(),
								  _threadIndividualResultStorage[_getThreadIndex()].begin() + exs.size());
}

template <typename T> std::vector<std::shared_ptr<typename Learn<T>::TrainingExample>> Learn<T>::_getNextExamples()
{
	std::unique_lock<std::mutex> queueLock(_exampleQueueMutex);
	size_t numRemainingExamples = _exampleQueue.size();

	// first check whether count has reached
	if ((_threadsTaken == _concurrency || numRemainingExamples == 0) && _threadsTaken != 0) {
		// wait for all the threads to have added to the results vector
		std::unique_lock<std::mutex> resultLock(_threadResultsMutex);
		_threadResultsConditionVariable.wait(resultLock, [&]() { // cv will get notified every time a result gets pushed
			// std::cout << "should not get callled alll the itme" << std::endl;
			return this->_threadsFinished == _threadsTaken; // so wait until all results have arrived
		});

		// all results have now arrived, so find average gradient and learn

		auto now = std::chrono::system_clock::now();
		double speed = (double)(_concurrency * _batchSizePerThread * (_epochIterations + 1)) /
					   std::chrono::duration<double>(now - _beginningOfEpochOrTest).count();

		// auto timeElapsed = std::chrono::duration<double, std::milli>(now - _lastIteration).count();
		// _lastIteration = now;

		CalculationResult average =
			_getAverageFromResults(_threadResults.begin(), _threadResults.begin() + _threadsFinished);

		if (!_testingMode) {
			_epochAccuracy = _epochAccuracy == 0.0 ? average._accuracy : 0.9 * _epochAccuracy + 0.1 * average._accuracy;

			// if (numRemainingExamples % 1000 == 0)
			std::cout << "epoch " << _trainingState._epochNumber << " cost " << average._cost << " accuracy "
					  << average._accuracy << " epoch accuracy " << _epochAccuracy << " remaining "
					  << numRemainingExamples << " speed " << speed << " single speed "
					  << average._singleCalculationSpeed << "\n";

			// apply
			for (size_t i = 0; i < _trainingState._biases.size(); ++i) {
				_trainingState._biases[i] -=
					_trainingState._learningRate * average._biasGradient[i] +
					_trainingState._learningRate * _momentum * _previousIterationAverage._biasGradient[i];
			}

			for (size_t i = 0; i < _trainingState._weights.size(); ++i) {
				_trainingState._weights[i] -=
					_trainingState._learningRate * average._weightGradient[i] +
					_trainingState._learningRate * _momentum * _previousIterationAverage._weightGradient[i];
			}

			std::vector<Mat> weightGradientCopy = average._weightGradient;
			std::vector<Vec> biasGradientCopy = average._biasGradient;

			_previousIterationAverage =
				CalculationResult({}, 0.0, std::move(weightGradientCopy), std::move(biasGradientCopy), 0.0, 0.0);
		} else {
			_testingAverage._cost =
				(_epochIterations * _testingAverage._cost + average._cost) / static_cast<double>(_epochIterations + 1);
			_testingAverage._accuracy = (_epochIterations * _testingAverage._accuracy + average._accuracy) /
										static_cast<double>(_epochIterations + 1);

			// if (numRemainingExamples % 1000 == 0)
			std::cout << "epoch " << _trainingState._epochNumber << " testing cost " << _testingAverage._cost
					  << " accuracy " << _testingAverage._accuracy << " remaining " << numRemainingExamples << " speed "
					  << speed << " single speed " << _testingAverage._singleCalculationSpeed << "\n";
		}

		// reset thread counters
		_threadsTaken = 0;
		_threadsFinished = 0;
		// _threadResults.clear();

		// toggle between testing and training modes
		if (numRemainingExamples == 0)
			_testingMode = !_testingMode;

		_epochIterations++;
	}

	// not all threads have been taken up
	_threadsTaken++;
	if (numRemainingExamples == 0 && !_testingMode) { // ended testing stage of epoch; new epoch

		if (!_initializing) { // do not call this on boot up
			_trainingState._testingReport.push_back(
				{_testingAverage._cost, _testingAverage._accuracy, _trainingState._epochNumber}); // add report
			saveLayerConnections();																  // save state
		}

		_previousIterationAverage = EmptyResult();
		_trainingState._epochNumber++;
		_epochAccuracy = 0.0;
		_epochIterations = 0;
		_exampleQueue =
			std::deque<std::shared_ptr<TrainingExample>>(_trainingExamples.begin(), _trainingExamples.end());
		// shuffle
		std::shuffle(_exampleQueue.begin(), _exampleQueue.end(), std::default_random_engine(0));
		numRemainingExamples = this->_trainingExamples.size();

		// use time-based learning schedule
		_trainingState._learningRate = _trainingState._learningRate / (1.0 + _decay * _trainingState._epochNumber);
		_beginningOfEpochOrTest = std::chrono::system_clock::now();

		_initializing = false;
	} else if (numRemainingExamples == 0 && _testingMode) { // ended training stage of epoch
		_testingAverage = EmptyResult();
		_epochIterations = 0;
		_exampleQueue = std::deque<std::shared_ptr<TrainingExample>>(_testingExamples.begin(), _testingExamples.end());
		numRemainingExamples = this->_testingExamples.size();
		_beginningOfEpochOrTest = std::chrono::system_clock::now();
	}

	// create list of next examples
	size_t exampleGroupSize = std::min(numRemainingExamples, _batchSizePerThread);
	std::vector<std::shared_ptr<TrainingExample>> nextExamples(_exampleQueue.begin(),
															   _exampleQueue.begin() + exampleGroupSize);
	_exampleQueue.erase(_exampleQueue.begin(), _exampleQueue.begin() + exampleGroupSize);

	return nextExamples;
}

template <typename T> void Learn<T>::_threadWorkerFunc(size_t epochs)
{
	while (_trainingState._epochNumber <= epochs) { // todo: make this variable
		auto nextExamples = _getNextExamples();
		auto results = _calculateMultiple(nextExamples);

		std::unique_lock<std::mutex> resultsLock(_threadResultsMutex);
		_threadResults[_threadsFinished++] = std::move(results);
		_threadResultsConditionVariable.notify_one();
	}
}

template <typename T> void Learn<T>::train(size_t epochs)
{
	_threadIndices.clear(); // reset thread indices

	std::vector<std::thread> threads;

	for (size_t t = 0; t < _concurrency; ++t) {
		threads.push_back(std::thread(&Learn::_threadWorkerFunc, this, epochs));
	}

	for (auto &t : threads) {
		t.join();
	}
}

template <typename T> void Learn<T>::saveLayerConnections()
{
	std::ofstream trainingStateFile(_saveFileLocation, std::ios::out | std::ios::trunc);
	trainingStateFile << std::flush;
	boost::archive::text_oarchive oa(trainingStateFile);
	oa &_trainingState;
}

template <typename T> void Learn<T>::loadLayerConnections()
{
	std::lock_guard<std::mutex> lock(_exampleQueueMutex);

	std::ifstream trainingStateFile(_saveFileLocation);

	boost::archive::text_iarchive ia(trainingStateFile);
	ia &_trainingState;

	std::cout << "Loaded training state.\n"
				 "Epochs Trained:\t"
			  << _trainingState._epochNumber - 1 << "\n"
			  << "Current Learning Rate:\t" << _trainingState._learningRate << "\nReport:";
	for (const auto &i : _trainingState._testingReport) {
		std::cout << "Epoch " << i._epoch << "\t Accuracy " << i._accuracy << "\t Cost " << i._cost << std::endl;
	}
}

template <typename T>
typename Learn<T>::TestingResult Learn<T>::predictIndividual(const std::shared_ptr<TrainingExample> &ex,
															 bool print) const
{
	CalculationResult calc = _calculateIndividual(ex);

	std::vector<double> stdOutputLayer(calc._outputLayer.data(), calc._outputLayer.data() + calc._outputLayer.size());

	T predicted = ex->interpretOutput(stdOutputLayer);
	T actual = ex->interpretOutput(ex->_expectedOutput);
	double confidence = ex->calculateConfidence(stdOutputLayer);

	TestingResult res{ex, predicted, actual, confidence};

	if (!print)
		return res;

	std::cout << res.ex->toString() << "\n"
			  << "Predicted: " << res.predictedInterpretation << "\n"
			  << "Actual: " << res.actualInterpretation << "\n"
			  << "Confidence: " << res.confidence << std::endl;

	return res;
}

template <typename T> size_t Learn<T>::_getThreadIndex() const
{
	std::thread::id id = std::this_thread::get_id();

	static size_t currentIndex = 0;
	if (_threadIndices.find(id) == _threadIndices.end()) {
		_threadIndices.insert(std::pair<std::thread::id, size_t>(id, currentIndex++));
	}

	return _threadIndices[id];
}

template class Learn<std::string>;
template class Learn<double>;
template class Learn<unsigned char>;