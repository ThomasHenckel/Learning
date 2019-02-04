#include "PID.h"
#include <cstdlib>
#include <iostream>

/**
 * TODO: Complete the PID class. You may add any additional desired functions.
 */

PID::PID() {}

PID::~PID() {}

void PID::Init(double Kp_, double Kd_,double Ki_) {
  /**
   * TODO: Initialize PID coefficients (and errors, if needed)
   */
  p[0] = Kp_;
  p[1] = Kd_;
  p[2] = Ki_;
  integral = 0;
  cte_prev = 0;
  state = 0;
  best_error = 0;
}

void PID::Test() {
  /**
   * TODO: Initialize PID coefficients (and errors, if needed)
   */
  if(best_error ==0)
  {
    best_error = error;
    std::cout << "better parameters found: " << p[0] << ":" << p[1] << ":" << p[2] <<std::endl;
  }

  std::cout << "error: : " << error<< "best_error" << best_error <<std::endl;
  if (state == 0){
    p[test_i] += dp[test_i];
    state = 1;
    std::cout << "testing Parameters up: " << p[0] << ":" << p[1] << ":" << p[2] <<std::endl;
    return;
  }
  if (state == 1){
    if (error < best_error){
      best_error = error;
      dp[test_i] *= 1.1;
      state = 0;
      test_i = (test_i+1)%3;
      std::cout << "better parameters found: " << p[0] << ":" << p[1] << ":" << p[2] <<std::endl;
    }
    else{
      p[test_i] -= 2 * dp[test_i];
      state = 2;
      std::cout << "testing Parameters: down " << p[0] << ":" << p[1] << ":" << p[2] <<std::endl;
      return;
    }
  }
  if(state == 2)
  {
    if (error < best_error){
      best_error = error;
      dp[test_i] *= 1.1;
      state = 0;
      test_i = (test_i+1)%3;
      std::cout << "better parameters found: " << p[0] << ":" << p[1] << ":" << p[2] <<std::endl;
    }
    else
    {
      p[test_i] += dp[test_i];
      dp[test_i] *= 0.9;
      state = 0;
      test_i = (test_i+1)%3;
      Test();
    }
  }
  return;
}


void PID::UpdateError(double cte) {
  /**
   * TODO: Update PID errors based on cte.
   */
  integral = integral + cte;

  p_error = -p[0] * cte;
  d_error = p[1] * (cte-cte_prev);
  i_error = p[2] * integral;

  cte_prev = cte;
  error = error + abs(cte);
}

double PID::TotalError() {
  /**
   * TODO: Calculate and return the total error
   */
  return p_error - d_error - i_error;  // total error calc
}