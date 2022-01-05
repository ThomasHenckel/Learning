#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
	VectorXd rmse(4);
	rmse << 0,0,0,0;
    
	//accumulate squared residuals
	VectorXd residual(4);
	for(int i=0; i < estimations.size(); ++i){
    residual = estimations[i] - ground_truth[i];
    residual = residual.array()*residual.array();
		rmse += residual;
	}

	//calculate the mean
	rmse = rmse/estimations.size();
    
	//calculate the squared root
  rmse = rmse.array().sqrt();

	return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
	MatrixXd Hj(3,4);
	//recover state parameters
	float px = x_state(0);
	float py = x_state(1);
	float vx = x_state(2);
	float vy = x_state(3);

	float pxy2 = px*px + py*py;
	//check division by zero
	if(fabs(pxy2) < 0.0001){
		cout << "CalculateJacobian () - Error - Division by Zero" << endl;
		return Hj;
	}
	
	float power = 1.5;
  //compute the Jacobian matrix
	Hj << px / sqrt(pxy2),py / sqrt(pxy2), 0, 0,
	      -py/pxy2, px/pxy2, 0, 0,
	      py*(vx*py-vy*px)/pow(pxy2,power), px*(vy*px-vx*py)/pow(pxy2,power), px/sqrt(pxy2), py/sqrt(pxy2);
	

	return Hj;
}
