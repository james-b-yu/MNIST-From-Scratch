#include "./linear.h"

Linear::Linear(const double &y, const double &m, const double &x, const double &c)
	: _y(y), _m(m), _x(x), _c(c), TrainingExample({m, x, c}, {y})
{
}

std::string Linear::toString() const
{
	std::stringstream sstr;
	sstr << "y = " << _y << "\n"
		 << "m = " << _m << "\n"
		 << "x = " << _x << "c =" << _c;

	return sstr.str();
}

std::string Linear::interpretOutput(const std::vector<double> &output) const
{
	std::stringstream sstr;
	sstr << _m << " * " << _x << " + " << _c << " approx " << output[0];
	return sstr.str();
}