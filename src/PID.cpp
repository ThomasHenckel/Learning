#include "PID.h"

/**
 * TODO: Complete the PID class. You may add any additional desired functions.
 */

PID::PID() {}

PID::~PID() {}

void PID::Init(double Kp_, double Ki_, double Kd_) {
  /**
   * TODO: Initialize PID coefficients (and errors, if needed)
   */
  Kp = Kp_;
  Kd = Kd_;
  Ki = Ki_;
  integral = 0;
  cte_prev = 0;

}

void PID::UpdateError(double cte) {
  /**
   * TODO: Update PID errors based on cte.
   */
  integral = integral + cte;

  p_error = -Kp * cte;
  i_error = Ki * integral;
  d_error = Kd * (cte-cte_prev);

  cte_prev = cte;
}

double PID::TotalError() {
  /**
   * TODO: Calculate and return the total error
   */
  return p_error - d_error - i_error;  // total error calc
}