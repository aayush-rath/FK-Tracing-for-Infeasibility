#include <Eigen/Dense>
#include <iostream>
#include <iomanip>

int main(int argc, char* argv[]) {
	int dim = std::stoi(argv[1]);
	Eigen::MatrixXd cartan(Eigen::MatrixXd::Identity(dim,dim));
	for (unsigned i = 1; i < dim; i++) {
		cartan(i-1, i) = -0.5;
		cartan(i, i-1) = -0.5;
	}

	Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes(cartan);
	Eigen::VectorXd sqrt_diag(dim);
	for (unsigned i = 0; i < dim; i++) sqrt_diag(i) = std::sqrt(saes.eigenvalues()[i]);

	Eigen::MatrixXd lower(Eigen::MatrixXd::Ones(dim,dim));
	lower = lower.triangularView<Eigen::Lower>();

	Eigen::MatrixXd result = (lower * saes.eigenvectors() * sqrt_diag.asDiagonal()).inverse();
	Eigen::MatrixXd result_inv = result.inverse();

	for (int i = 0; i < dim; i++) {
		for (int j = 0; j < dim; j++) {
			std::cout << "Lambda[" << i << "][" << j << "] = " << std::fixed << std::setprecision(10) << result(i, j) << ";\t";
		}
		std::cout << std::endl;
	}

	std::cout << std::endl << "Inverse: " << std::endl;

	for (int i = 0; i < dim; i++) {
		for (int j = 0; j < dim; j++) {
			std::cout << "Lambda_inv[" << i << "][" << j << "] = "  << std::fixed << std::setprecision(10) << result_inv(i, j) << ";\t";
		}
		std::cout << std::endl;
	}

	Eigen::VectorXd point(dim);
	point << 1.06753, -0.533766, 0.0, -0.0376776, 0.0;
	std::cout << std::endl;
	std::cout << "Orig: ";
	for (int i = 0; i < dim; i++) std::cout << point[i] << " ";
	std::cout << std::endl;
	Eigen::VectorXd mult_point = result_inv * point;
	mult_point *= 100.0;
	std::cout << std::endl << "Point: ";
	for (int i = 0; i < dim; i++) std::cout << mult_point[i] << " ";
	std::cout << std::endl;
	Eigen::VectorXd pointo(dim);
	pointo = mult_point / 100.0;
	pointo = result * pointo;
	std::cout << "Pointo: ";
	for (int i = 0; i < dim; i++) std::cout << pointo(i) << " ";
	std::cout << std::endl;
}
