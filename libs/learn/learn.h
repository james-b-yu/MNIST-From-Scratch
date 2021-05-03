#pragma once
// #define EIGEN_USE_MKL_ALL
#include <atomic>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/serialization/split_member.hpp>
#include <boost/serialization/vector.hpp>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <eigen3/Eigen/Dense>
#include <fstream>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> Mat;
typedef Eigen::Matrix<double, Eigen::Dynamic, 1> Vec;

struct TestingReportEntry {
	double _cost;
	double _accuracy;
	size_t _epoch;
};

struct TrainingState {
	std::vector<Mat> _weights;
	std::vector<Vec> _biases;
	double _learningRate;
	size_t _epochNumber;
	std::vector<TestingReportEntry> _testingReport;
};

template <typename T> class Learn
{
  public:
	struct TrainingExample {
	  private:
	  public:
		std::vector<double> _input;
		std::vector<double> _expectedOutput;

		virtual std::string toString() const;
		virtual double calculateAbsoluteAccuracy(const std::vector<double> &calculatedOutput) const;
		virtual double calculateConfidence(const std::vector<double> &calculatedOutput) const;
		virtual T interpretOutput(const std::vector<double> &output) const;

		TrainingExample(const std::vector<double> &input, const std::vector<double> expectedOutput);
		TrainingExample(std::vector<double> &&input, std::vector<double> &&expectedOutput);
		TrainingExample(TrainingExample &&t) = default;
	};

	struct TestingResult {
		std::shared_ptr<TrainingExample> ex;
		T predictedInterpretation;
		T actualInterpretation;
		double confidence;
	};

	struct CalculationResult {
		Vec _outputLayer;
		double _cost;
		std::vector<Mat> _weightGradient;
		std::vector<Vec> _biasGradient;
		double _accuracy;
		double _singleCalculationSpeed;

		CalculationResult(Vec &&outputLayer, double &&cost, std::vector<Mat> &&weightGradient,
						  std::vector<Vec> &&biasGradient, double accuracy, double singleCalculationSpeed);
	};

	CalculationResult EmptyResult();

	Learn(size_t numInputNodes, size_t numHiddenLayers, size_t numHiddenLayerNodes, size_t numOutputNodes,
		  double learningRate, double decay, double momentum,
		  std::vector<std::shared_ptr<TrainingExample>> &&trainingExamples,
		  std::vector<std::shared_ptr<TrainingExample>> &&testingExamples, size_t concurrency,
		  size_t batchSizePerThread, std::string saveFileLocation);
	Learn(const Learn &) = delete; // disable copy constructor

	void setRandomLayerConnections();
	void train(size_t epochs = 10);

	void saveLayerConnections();
	void loadLayerConnections();

	TestingResult predictIndividual(const std::shared_ptr<TrainingExample> &ex, bool print = true) const;

  private:
	size_t _numInputNodes;
	size_t _numHiddenLayers;
	size_t _numHiddenLayerNodes;
	size_t _numOutputNodes;
	size_t _numLayers;
	double _decay;
	double _momentum;

	// std::vector<Mat> _weights;
	// std::vector<Vec> _biases;

	std::vector<Mat> _initializeLikeWeights() const;
	std::vector<Vec> _initializeLikeBiases() const;
	std::vector<Vec> _initializeLikeLayers(const std::vector<double> &in) const;

	static double _scalingFunc(const double &v);
	static double _dScalingFunc(const double &v);
	static double _scalingFuncOutput(const double &v);
	static double _dScalingFuncOutput(const double &v);

	static double _costFunc(const double &c, const double &e);
	static double _dCostFunc(const double &c, const double &e);

	// static double _costFunc(const Vec &c, const Vec &e);
	CalculationResult _getAverageFromResults(const std::vector<CalculationResult> &res) const;

	// THREADING!!
	std::mutex _exampleQueueMutex;
	std::mutex _threadResultsMutex;

	double _epochAccuracy = 0.0;
	size_t _epochIterations = 0;

	std::vector<std::shared_ptr<TrainingExample>> _getNextExamples();
	size_t _threadsTaken = 0;

	void _threadWorkerFunc(size_t epochs);

	std::deque<std::shared_ptr<TrainingExample>> _exampleQueue;
	std::vector<CalculationResult> _threadResults;
	CalculationResult _previousIterationAverage = CalculationResult({}, 0.0, {}, {}, 0.0, 0.0);
	std::condition_variable _threadResultsConditionVariable;

	std::chrono::system_clock::time_point _beginningOfEpochOrTest = std::chrono::system_clock::now();
	bool _testingMode = false;

	std::vector<std::shared_ptr<TrainingExample>> _trainingExamples;
	std::vector<std::shared_ptr<TrainingExample>> _testingExamples;
	CalculationResult _testingAverage = CalculationResult({}, 0.0, {}, {}, 0.0, 0.0);

	size_t _concurrency = 2;
	size_t _batchSizePerThread = 20;

	TrainingState _trainingState;
	std::string _saveFileLocation;

	CalculationResult _calculateIndividual(std::shared_ptr<TrainingExample> ex) const;
	CalculationResult _calculateMultiple(std::vector<std::shared_ptr<TrainingExample>> exs) const;
};

namespace boost
{
namespace serialization
{
template <class Archive> void serialize(Archive &ar, TrainingState &lc, const unsigned int)
{
	ar &lc._learningRate;
	ar &lc._epochNumber;
	ar &lc._biases;
	ar &lc._weights;
	ar &lc._testingReport;
}

template <class Archive> void serialize(Archive &ar, TestingReportEntry &re, const unsigned int)
{
	ar &re._accuracy;
	ar &re._cost;
	ar &re._epoch;
}

template <class Archive, typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
void save(Archive &ar, const Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols> &mat,
		  const unsigned int)
{
	ar &mat.rows();
	ar &mat.cols();
	ar &boost::serialization::make_array(mat.data(), mat.rows() * mat.cols());
}

template <class Archive, typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
void load(Archive &ar, Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols> &mat, const unsigned int)
{
	int rows;
	int cols;
	ar &rows;
	ar &cols;
	mat.resize(rows, cols);
	ar &boost::serialization::make_array(mat.data(), rows * cols);
}

template <class Archive, typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
void serialize(Archive &ar, Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols> &mat,
			   const unsigned int version)
{
	boost::serialization::split_free(ar, mat, version);
}

} // namespace serialization
} // namespace boost