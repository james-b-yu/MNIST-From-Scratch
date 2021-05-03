#include "../learn/learn.h"

class Linear : public Learn<std::string>::TrainingExample
{
  private:
	double _x;
	double _m;
	double _c;
	double _y;

  public:
	virtual std::string toString() const;

	virtual double calculateAbsoluteAccuracy(const std::vector<double> &calculatedOutput) const
	{
		return (calculatedOutput[0] - _y) / _y;
	};

	virtual double calculateConfidence(const std::vector<double> &calculatedOutput) const
	{
		return 0.0;
	};

	virtual std::string interpretOutput(const std::vector<double> &output) const;

	Linear(const double &y, const double &m, const double &x, const double &c);
};