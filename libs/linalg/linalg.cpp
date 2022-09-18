#include "linalg.h"
#include <iostream>

LinAlg::Vector::Vector(const Vector &v)
{
	_size = v._size;
	_array = v._array;
}

LinAlg::Vector::Vector(size_t size)
{
	_array = std::vector<double>(size); // should not be able to change size afterwards
	_size = size;
}

LinAlg::Vector::Vector(std::initializer_list<double> l) : _array(l)
{
	_size = l.size();
}

LinAlg::Vector::Vector(std::vector<double> l) : _array(l)
{
	_size = l.size();
}

double &LinAlg::Vector::operator[](size_t index)
{
	return _array[index];
}

const double &LinAlg::Vector::operator[](size_t index) const
{
	return _array[index];
}

const size_t &LinAlg::Vector::size() const
{
	return _size;
}

void LinAlg::Vector::operator=(const Vector &a)
{
	_array = a._array;
	_size = a._size;
}

LinAlg::Matrix::Matrix(const Matrix &m)
{
	_matrix = m._matrix;
	_size = m._size;
}

LinAlg::Matrix::Matrix(size_t m, size_t n)
{
	_matrix = {};
	for (size_t i = 0; i < m; ++i)
		_matrix.push_back(Vector(n));

	_size = std::make_pair<size_t, size_t>(std::move(m), std::move(n));
}

LinAlg::Matrix::Matrix(std::initializer_list<Vector> m) : _matrix(m)
{
	// check if empty
	if (!m.size())
		throw std::invalid_argument("m must not be empty");
	// check that all arrays have the same length
	const size_t n = m.begin()->size();
	for (const Vector &a : m) {
		if (a.size() != n) {
			throw std::invalid_argument("all arrays in m must have same size n");
		}
	}

	_size = std::pair<size_t, size_t>(m.size(), n);
}

LinAlg::Vector &LinAlg::Matrix::operator[](size_t m)
{
	return _matrix[m];
}

const LinAlg::Vector &LinAlg::Matrix::operator[](size_t m) const
{
	return _matrix[m];
}

void LinAlg::Matrix::operator=(const LinAlg::Matrix &r)
{
	_matrix = r._matrix;
	_size = r._size;
}

const std::pair<size_t, size_t> &LinAlg::Matrix::size() const
{
	return _size;
}

// todo: overload the matrix multiply function instead of casting
LinAlg::Vector::operator Matrix() const
{
	Matrix result(_size, 1);
	for (size_t i = 0; i < _size; ++i) {
		result[i][0] = _array[i];
	};

	return result;
}

LinAlg::Matrix::operator Vector() const
{
	// if not a column matrix, throw error
	if (_size.first != 1 && _size.second != 1) {
		throw std::invalid_argument("must be 1*n or m*1");
	}

	size_t vectorSize = std::max(_size.first, _size.second);

	Vector result(vectorSize);

	if (_size.first == 1)
		result = _matrix[0];
	else {
		for (size_t i = 0; i < vectorSize; ++i)
			result[i] = _matrix[i][0];
	}

	return result;
}

// for ostream

std::ostream &operator<<(std::ostream &s, const LinAlg::Vector &a)
{
	for (size_t i = 0; i < a.size(); ++i) {
		s << a[i];
		if (i + 1 != a.size()) {
			s << "\t"; // add tab but not after the final one
		}
	}

	return s;
}

std::ostream &operator<<(std::ostream &s, const LinAlg::Matrix &m)
{
	for (size_t i = 0; i < m.size().first; ++i) {
		s << m[i];
		if (i + 1 != m.size().first) {
			s << "\n"; // add tab but not after the final one
		}
	}

	return s;
}

// functions

double LinAlg::dot(const Vector &l, const Vector &r)
{
	// check that sizes are the same
	if (l.size() != r.size()) {
		throw std::invalid_argument("l and r must have identical sizes");
	}

	double sum = 0.0;

	for (size_t i = 0; i < l.size(); ++i) {
		sum += l[i] * r[i];
	}

	return sum;
}

double operator*(const LinAlg::Vector &l, const LinAlg::Vector &r)
{
	return LinAlg::dot(l, r);
}

LinAlg::Vector operator+(const LinAlg::Vector &l, const LinAlg::Vector &r)
{
	const size_t size = l.size();
	if (size != r.size()) {
		throw std::invalid_argument("both vectors must have the same size");
	}

	LinAlg::Vector result(size);
	for (size_t i = 0; i < size; ++i) {
		result[i] = l[i] + r[i];
	}

	return result;
}

LinAlg::Matrix operator+(const LinAlg::Matrix &l, const LinAlg::Matrix &r)
{
	const auto size = l.size();
	if (size != r.size()) {
		throw std::invalid_argument("both matrices must have the same size");
	}

	LinAlg::Matrix result(size.first, size.second);
	for (size_t i = 0; i < size.first; ++i) {
		for (size_t j = 0; j < size.second; ++j) {
			result[i][j] = l[i][j] + r[i][j];
		}
	}

	return result;
}

LinAlg::Vector operator*(const LinAlg::Vector &l, const double &r)
{
	auto result = l;
	for (size_t i = 0; i < result.size(); ++i) {
		result[i] *= r;
	}
	return result;
}
LinAlg::Vector operator*(const double &l, const LinAlg::Vector &r)
{
	return r * l;
}
LinAlg::Vector operator/(const LinAlg::Vector &l, const double &r)
{
	auto result = l;
	for (size_t i = 0; i < result.size(); ++i) {
		result[i] /= r;
	}
	return result;
}

LinAlg::Matrix operator*(const LinAlg::Matrix &l, const double &r)
{
	auto result = l;
	for (size_t i = 0; i < result.size().first; ++i) {
		for (size_t j = 0; j < result.size().second; ++j) {
			result[i][j] *= r;
		}
	}
	return result;
}
LinAlg::Matrix operator*(const double &l, const LinAlg::Matrix &r)
{
	return r * l;
}
LinAlg::Matrix operator/(const LinAlg::Matrix &l, const double &r)
{
	auto result = l;
	for (size_t i = 0; i < result.size().first; ++i) {
		for (size_t j = 0; j < result.size().second; ++j) {
			result[i][j] /= r;
		}
	}
	return result;
}

LinAlg::Matrix LinAlg::mmul(const Matrix &l, const Matrix &r)
{
	const auto &lSize = l.size();
	const auto &rSize = r.size();

	// l right must equal r left
	if (lSize.second != rSize.first) {
		throw std::invalid_argument("n size of first matrix must equal m size of second");
	}

	Matrix result(lSize.first, rSize.second);
	for (size_t m = 0; m < lSize.first; ++m) {
		for (size_t n = 0; n < rSize.second; ++n) {
			double v = 0.0;
			for (size_t o = 0; o < lSize.second; ++o) {
				v += l[m][o] * r[o][n];
			}

			result[m][n] = v;
		}
	}

	return result;
}

LinAlg::Matrix LinAlg::mpow(const Matrix &m, const unsigned long &p)
{
	// check that it is a square matrix
	const auto &mSize = m.size();
	if (mSize.first != mSize.second) {
		throw std::invalid_argument("m must be a square matrix");
	}

	// create diagonal matrix
	Matrix result(mSize.first, mSize.first);
	for (size_t i = 0; i < mSize.first; ++i) {
		result[i][i] = 1;
	}

	for (size_t i = 0; i < p; ++i) {
		result = LinAlg::mmul(result, m);
	}

	return result;
}

LinAlg::Matrix LinAlg::mtrans(const LinAlg::Matrix &m)
{
	LinAlg::Matrix result(m.size().second, m.size().first);

	for (size_t i = 0; i < m.size().first; ++i) {
		for (size_t j = 0; j < m.size().second; ++j) {
			result[j][i] = m[i][j];
		}
	}

	return result;
}

LinAlg::Matrix LinAlg::hprod(const LinAlg::Matrix &l, const LinAlg::Matrix &r)
{
	if (l.size() != r.size())
		throw std::invalid_argument("l and r matrices must have same sizes");

	LinAlg::Matrix result(l.size().first, l.size().second);

	for (size_t i = 0; i < l.size().first; ++i) {
		for (size_t j = 0; j < l.size().second; ++j) {
			result[i][j] = l[i][j] * r[i][j];
		}
	}

	return result;
}

LinAlg::Vector LinAlg::hprod(const LinAlg::Vector &l, const LinAlg::Vector &r)
{
	if (l.size() != r.size())
		throw std::invalid_argument("l and r vectors must have same sizes");

	LinAlg::Vector result(l.size());

	for (size_t i = 0; i < l.size(); ++i) {
		result[i] = l[i] * r[i];
	}

	return result;
}