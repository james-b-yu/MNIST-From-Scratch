#pragma once
#include <eigen3/Eigen/Dense>
#include <memory>
#include <vector>

class Learn
{
  public:
	typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> Mat;
	typedef Eigen::Matrix<double, Eigen::Dynamic, 1> Vec;

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

	struct Result {
		Vec _outputLayer;
		double _cost;
		std::vector<Mat> _weightGradient;
		std::vector<Vec> _biasGradient;
		double _accuracy;

		Result(Vec &&outputLayer, double &&cost, std::vector<Mat> &&weightGradient, std::vector<Vec> &&biasGradient,
			   double accuracy);
	};

	Learn(size_t numInputNodes, size_t numHiddenLayers, size_t numHiddenLayerNodes, size_t numOutputNodes,
		  std::vector<std::shared_ptr<TrainingExample>> &&trainingExamples, double learningRate);
	Learn(const Learn &) = delete; // disable copy constructor

	void _setRandomWeightsAndBiases();

	Result _calculateIndividual(std::shared_ptr<TrainingExample> ex) const;

	std::vector<std::shared_ptr<TrainingExample>> _trainingExamples;

  private:
	size_t _numInputNodes;
	size_t _numHiddenLayers;
	size_t _numHiddenLayerNodes;
	size_t _numOutputNodes;
	size_t _numLayers;
	double _learningRate;

	std::vector<Mat> _weights;
	std::vector<Vec> _biases;

	std::vector<Mat> _initializeLikeWeights() const;
	std::vector<Vec> _initializeLikeBiases() const;
	std::vector<Vec> _initializeLikeLayers(const std::vector<double> &in) const;

	static double _scalingFunc(const double &v);
	static double _dScalingFunc(const double &v);

	static double _costFunc(const double &c, const double &e);
	static double _dCostFunc(const double &c, const double &e);

	static double _costFunc(const Vec &c, const Vec &e);
	Result _getAverageFromResults(const std::vector<Result> &res) const;
};