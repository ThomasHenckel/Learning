#ifndef PID_H
#define PID_H

class PID {
 public:
  /**
   * Constructor
   */
  PID();

  /**
   * Destructor.
   */
  virtual ~PID();

  /**
   * Initialize PID.
   * @param (Kp_, Ki_, Kd_) The initial PID coefficients
   */
  void Init(double Kp_, double Kd_,double Ki_);

    /**
   * Initialize PID.
   * @param (Kp_, Ki_, Kd_) The initial PID coefficients
   */
  void Test();

  /**
   * Update the PID error variables given cross track error.
   * @param cte The current cross track error
   */
  void UpdateError(double cte);

  /**
   * Calculate the total PID error.
   * @output The total PID error
   */
  double TotalError();

  

 private:
  /**
   * PID Errors
   */
  double p_error;
  double i_error;
  double d_error;
  double integral;
  double cte_prev;

  /**
   * PID Coefficients
   */ 
  double p[3] = {0,0,0};

  double dp[3] = {0.02, 0.1, 0.001};

  int test_i=0;
  double error=0;
  double best_error=0;
  int state = 0;
};

#endif  // PID_H