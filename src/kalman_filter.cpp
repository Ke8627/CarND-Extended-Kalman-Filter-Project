#include <cmath>
#include <iostream>
#include "kalman_filter.h"
#include "tools.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in)
{
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
}

void KalmanFilter::Predict(const MatrixXd& Q) {
  /**
    * predict the state
  */
  x_ = F_ * x_;
  MatrixXd Ft = F_.transpose();
  P_ = F_ * P_ * Ft + Q;
}

void KalmanFilter::Update(const VectorXd &z, const MatrixXd &H, const MatrixXd &R) {
  /**
    * update the state by using Kalman Filter equations
  */
  VectorXd z_pred = H * x_;
  VectorXd y = z - z_pred;
  UpdateGeneralized(y, H, R);
}

void NormalizePhi(VectorXd& y)
{
  static const double two_pi = 2 * M_PI;

  auto& phi = y[1];
  while (phi > M_PI)
  {
    phi -= two_pi;
  }
  while (phi < -M_PI)
  {
    phi += two_pi;
  }
}

void KalmanFilter::UpdateEKF(const VectorXd &z, const MatrixXd &R) {
  /**
    * update the state by using Extended Kalman Filter equations
  */
  VectorXd y = z - Tools::ConvertCartesianToPolar(x_);

  NormalizePhi(y);

  MatrixXd Hj = Tools::CalculateJacobian(x_);

  UpdateGeneralized(y, Hj, R);
}

void KalmanFilter::UpdateGeneralized(const Eigen::VectorXd &y, const Eigen::MatrixXd &H, const Eigen::MatrixXd &R)
{
  MatrixXd Ht = H.transpose();
  MatrixXd S = H * P_ * Ht + R;
  MatrixXd Si = S.inverse();
  MatrixXd K = P_ * Ht * Si;

  x_ = x_ + (K * y);
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K * H) * P_;
}
