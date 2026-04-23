#include <Eigen/Dense>
#include <iostream>

int main(int argc, char* argv[]) {
	Eigen::MatrixXd cartan(Eigen::MatrixXd::Identity(4,4));
	for (unsigned i = 1; i < 4; i++) {
		cartan(i-1, i) = -0.5;
		cartan(i, i-1) = -0.5;
	}

	Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes(cartan);
	Eigen::VectorXd sqrt_diag(4);
	for (unsigned i = 0; i < 4; i++) sqrt_diag(i) = std::sqrt(saes.eigenvalues()[i]);

	Eigen::MatrixXd lower(Eigen::MatrixXd::Ones(4,4));
	lower = lower.triangularView<Eigen::Lower>();

	Eigen::MatrixXd result = (lower * saes.eigenvectors() * sqrt_diag.asDiagonal()).inverse();
	Eigen::MatrixXd result_inv = result.inverse();

	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			std::cout << result(i, j) << " ";
		}
		std::cout << std::endl;
	}

	std::cout << std::endl << "Inverse: " << std::endl;

	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			std::cout << result_inv(i, j) << " ";
		}
		std::cout << std::endl;
	}

	Eigen::VectorXd point(4);
	point << 1.06753, -0.533766, 0.0, -0.0376776;
	std::cout << std::endl;
	std::cout << "Orig: ";
	for (int i = 0; i < 4; i++) std::cout << point[i] << " ";
	std::cout << std::endl;
	Eigen::VectorXd mult_point = result_inv * point;
	mult_point *= 100.0;
	std::cout << std::endl << "Point: ";
	for (int i = 0; i < 4; i++) std::cout << mult_point[i] << " ";
	std::cout << std::endl;
	Eigen::VectorXd pointo(4);
	pointo = mult_point / 100.0;
	pointo = result * pointo;
	std::cout << "Pointo: ";
	for (int i = 0; i < 4; i++) std::cout << pointo(i) << " ";
	std::cout << std::endl;
}
