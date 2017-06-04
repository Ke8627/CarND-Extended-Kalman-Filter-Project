#include "FusionEKF.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/*
 * Constructor.
 */
FusionEKF::FusionEKF() {
  is_initialized_ = false;

  previous_timestamp_ = 0;

  // initializing matrices
  R_laser_ = MatrixXd(2, 2);
  R_radar_ = MatrixXd(3, 3);
  H_laser_ = MatrixXd(2, 4);

  // Measurement covariance matrix - laser
  R_laser_ << 0.0225, 0,
              0, 0.0225;

  // Measurement covariance matrix - radar
  R_radar_ << 0.09, 0, 0,
              0, 0.0009, 0,
              0, 0, 0.09;

  /**
    * Finish initializing the FusionEKF.
    * Set the process and measurement noises
  */

  H_laser_ << 1, 0, 0, 0,
              0, 1, 0, 0;

}

/**
* Destructor.
*/
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {


  /*****************************************************************************
   *  Initialization
   ****************************************************************************/
  if (!is_initialized_) {
    /**
      * Initialize the state ekf_.x_ with the first measurement.
      * Create the covariance matrix.
      * Remember: you'll need to convert radar from polar to cartesian coordinates.
    */
    // first measurement
    cout << "EKF: " << endl;

    MatrixXd F(4, 4);
    F << 1, 0, 1, 0,
         0, 1, 0, 1,
         0, 0, 1, 0,
         0, 0, 0, 1;

    // state covariance matrix P

    MatrixXd P(4, 4);

    P << 1, 0, 0, 0,
         0, 1, 0, 0,
         0, 0, 1000, 0,
         0, 0, 0, 1000;

    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR)
    {
      // Convert radar from polar to cartesian coordinates and initialize state.
      VectorXd x = Tools::ConvertPolarToCartesian(measurement_pack.raw_measurements_);

      ekf_.Init(x, P, F);
    }
    else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER)
    {
      // Initialize state.
      VectorXd x(4);
      x << measurement_pack.raw_measurements_(0), measurement_pack.raw_measurements_[1], 0, 0;

      ekf_.Init(x, P, F);
    }

    // Done initializing, no need to predict or update
    is_initialized_ = true;

    previous_timestamp_ = measurement_pack.timestamp_;

    return;
  }

  /*****************************************************************************
   *  Prediction
   ****************************************************************************/

  static const double c_microsecondsPerSecond = 1000000;

  // Timestamps from measurement_pack are in microseconds.
  auto dt = (measurement_pack.timestamp_ - previous_timestamp_) / c_microsecondsPerSecond;

  // Update the state transition matrix F according to the new elapsed time.
  // Time is measured in seconds.
  ekf_.F_(0, 2) = dt;
  ekf_.F_(1, 3) = dt;

  // Update the process noise covariance matrix Q.
  // Use noise_ax = 9 and noise_ay = 9 for your Q matrix.
  //
  const static float noise_ax = 9;
  const static float noise_ay = 9;

  float dt2 = dt * dt;
  float dt3 = dt2 * dt / 2;
  float dt4 = dt2 * dt2 / 4;

  MatrixXd Q(4, 4);

  Q << (dt4 * noise_ax), 0, (dt3 * noise_ax), 0,
       0, (dt4 * noise_ay), 0, (dt3 * noise_ay),
       (dt3 * noise_ax), 0, (dt2 * noise_ax), 0,
       0, (dt3 * noise_ay), 0, (dt2 * noise_ay);

  ekf_.Predict(Q);

  /*****************************************************************************
   *  Update
   ****************************************************************************/

  /**
     * Use the sensor type to perform the update step.
     * Update the state and covariance matrices.
   */

  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    // Radar updates
    ekf_.UpdateEKF(measurement_pack.raw_measurements_, R_radar_);
  } else {
    // Laser updates
    ekf_.Update(measurement_pack.raw_measurements_, H_laser_, R_laser_);
  }

  previous_timestamp_ = measurement_pack.timestamp_;

  // print the output
  cout << "x_ = " << ekf_.x_ << endl;
  cout << "P_ = " << ekf_.P_ << endl;
}
