#include "../linalg/linalg.h"
#include <atomic>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#pragma once
class Learn
{
  public:
	struct Result {
		LinAlg::Vector _outputLayer;
		double _cost;
		std::vector<LinAlg::Matrix> _weightGradient;
		std::vector<LinAlg::Vector> _biasGradient;
		double _accuracy;

		Result(LinAlg::Vector &&_outputLayer, double &&cost, std::vector<LinAlg::Matrix> &&weightGradient,
			   std::vector<LinAlg::Vector> &&biasGradient, double accuracy);
	};

	struct TrainingExample {
	  private:
	  public:
		std::vector<double> _input;
		std::vector<double> _expectedOutput;

		virtual std::string toString()
		{
			return "";
		}

		TrainingExample(const std::vector<double> &input, const std::vector<double> expectedOutput);
		TrainingExample(std::vector<double> &&input, std::vector<double> &&expectedOutput);
		TrainingExample(TrainingExample &&t) = default;
	};

	void _setRandomWeightsAndBiases();
	Result _calculateIndividual(std::shared_ptr<TrainingExample> ex) const;
	std::vector<Result> _calculateMultiple(std::vector<std::shared_ptr<TrainingExample>> exs) const;

	Learn(size_t numInputNodes, size_t numHiddenLayers, size_t numHiddenLayerNodes, size_t numOutputNodes,
		  std::vector<std::shared_ptr<TrainingExample>> &&trainingExamples, double learningRate);
	Learn(const Learn &) = delete; // disable copy constructor

	void train(size_t epochs = 10);
	std::vector<std::shared_ptr<TrainingExample>> _trainingExamples;
	Result _getAverageFromResults(const std::vector<Result> &res) const;

  private:
	size_t _numInputNodes;
	size_t _numLayers;
	size_t _numHiddenLayers;
	size_t _numHiddenLayerNodes;
	size_t _numOutputNodes;
	double _learningRate;

	std::vector<LinAlg::Matrix> _weights;
	std::vector<LinAlg::Vector> _biases;

	LinAlg::Vector _scalingFunc(const LinAlg::Vector &v) const;
	double _scalingFunc(const double &v) const;
	LinAlg::Vector _scalingFuncOutput(const LinAlg::Vector &v) const;
	double _scalingFuncOutput(const double &v) const;

	double _dScalingFunc(const double &v) const;
	LinAlg::Vector _dScalingFunc(const LinAlg::Vector &v) const;
	double _dScalingFuncOutput(const double &v) const;
	LinAlg::Vector _dScalingFuncOutput(const LinAlg::Vector &v) const;

	double _costFunc(const double &calculated, const double &expected) const;
	double _dCostFunc(const double &calculated, const double &expected) const;
	LinAlg::Vector _dCostFunc(const LinAlg::Vector &calculated, const LinAlg::Vector &expected) const;
	double _costFunc(const LinAlg::Vector &calculated, const LinAlg::Vector &expected) const;

	void _initializeLikeWeights(std::vector<LinAlg::Matrix> &weights) const;
	void _initializeLikeBiases(std::vector<LinAlg::Vector> &biases) const;
	void _initializeLikeLayers(std::vector<LinAlg::Vector> &layers, const LinAlg::Vector &in) const;

	// THREADING!!
	std::mutex _exampleQueueMutex;
	std::mutex _threadResultsMutex;

	std::vector<std::shared_ptr<TrainingExample>> _getNextExamples();
	size_t _threadsTaken = 0;
	size_t _concurrency = 2;

	size_t _epochNumber = 0;
	void _threadWorkerFunc(size_t epochs);

	std::deque<std::shared_ptr<TrainingExample>> _exampleQueue;
	std::vector<Result> _threadResults;
	std::condition_variable _threadResultsConditionVariable;
};