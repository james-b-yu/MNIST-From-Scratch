#include <initializer_list>
#include <iostream>
#include <vector>

#pragma once

namespace LinAlg
{
class Matrix;
class Vector
{
  private:
	std::vector<double> _array;
	size_t _size;

  public:
	Vector(const Vector &v);
	Vector(size_t size);
	Vector(std::initializer_list<double> l);
	Vector(std::vector<double> l);

	double &operator[](size_t index);
	const double &operator[](size_t index) const;

	void operator=(const Vector &a);
	const size_t &size() const;

	operator Matrix() const;
}; // class Vector

class Matrix
{
  private:
	std::vector<Vector> _matrix;
	std::pair<size_t, size_t> _size;

  public:
	Matrix(size_t m, size_t n);
	Matrix(std::initializer_list<Vector> m);
	Matrix(const Matrix &m);

	Vector &operator[](size_t m);
	const Vector &operator[](size_t m) const;

	void operator=(const Matrix &m);
	const std::pair<size_t, size_t> &size() const;

	operator Vector() const;
};

double dot(const Vector &l, const Vector &r);

Matrix mmul(const Matrix &l, const Matrix &r);

Matrix mtrans(const Matrix &m);

Matrix hprod(const Matrix &l, const Matrix &r);

Vector hprod(const Vector &l, const Vector &r);

Matrix mpow(const Matrix &m, const unsigned long &p);
} // namespace LinAlg

std::ostream &operator<<(std::ostream &s, const LinAlg::Matrix &m);
std::ostream &operator<<(std::ostream &s, const LinAlg::Vector &a);
double operator*(const LinAlg::Vector &l, const LinAlg::Vector &r);
LinAlg::Vector operator+(const LinAlg::Vector &l, const LinAlg::Vector &r);
LinAlg::Vector operator*(const LinAlg::Vector &l, const double &r);
LinAlg::Vector operator*(const double &r, const LinAlg::Vector &l);
LinAlg::Vector operator/(const LinAlg::Vector &l, const double &r);

LinAlg::Matrix operator+(const LinAlg::Matrix &l, const LinAlg::Matrix &r);
LinAlg::Matrix operator*(const LinAlg::Matrix &l, const double &r);
LinAlg::Matrix operator*(const double &r, const LinAlg::Matrix &l);
LinAlg::Matrix operator/(const LinAlg::Matrix &l, const double &r);