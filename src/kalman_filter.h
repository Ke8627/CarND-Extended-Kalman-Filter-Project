#ifndef KALMAN_FILTER_H_
#define KALMAN_FILTER_H_
#include "Eigen/Dense"

class KalmanFilter {
public:

  // state vector
  Eigen::VectorXd x_;

  // state covariance matrix
  Eigen::MatrixXd P_;

  // state transition matrix
  Eigen::MatrixXd F_;

  /**
   * Constructor
   */
  KalmanFilter();

  /**
   * Destructor
   */
  virtual ~KalmanFilter();

  /**
   * Init Initializes Kalman filter
   * @param x_in Initial state
   * @param P_in Initial state covariance
   * @param F_in Transition matrix
   */
  void Init(Eigen::VectorXd &x_in,
            Eigen::MatrixXd &P_in,
            Eigen::MatrixXd &F_in);

  /**
   * Prediction Predicts the state and the state covariance
   * using the process model
   * @param Q Process covariance matrix
   */
  void Predict(const Eigen::MatrixXd &Q);

  /**
   * Updates the state by using standard Kalman Filter equations
   * @param z The measurement at k+1
   * @param H Measurement matrix
   * @param R Measurement covariance matrix
   */
  void Update(const Eigen::VectorXd &z,
              const Eigen::MatrixXd &H,
              const Eigen::MatrixXd &R);

  /**
   * Updates the state by using Extended Kalman Filter equations
   * @param z The measurement at k+1
   * @param R Measurement covariance matrix
   */
  void UpdateEKF(const Eigen::VectorXd &z,
                 const Eigen::MatrixXd &R);

private:

  void UpdateGeneralized(const Eigen::VectorXd &z,
                         const Eigen::MatrixXd &H,
                         const Eigen::MatrixXd &R);

};

#endif /* KALMAN_FILTER_H_ */
