#include <cmath>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) 
{
  VectorXd rmse(4);

  rmse << 0,0,0,0;

  // Check the validity of the inputs.
  if (estimations.size() != ground_truth.size() || estimations.size() == 0)
  {
    throw std::runtime_error("Invalid estimation or ground_truth data");
  }

  // Accumulate squared residuals.
  for (unsigned int i=0; i < estimations.size(); ++i) 
  {
    VectorXd residual = estimations[i] - ground_truth[i];

    // Coefficient-wise multiplication.
    residual = residual.array()*residual.array();

    rmse += residual;
  }

  // Calculate the mean.
  rmse = rmse/estimations.size();

  rmse = rmse.array().sqrt();

  return rmse;
}

// This implementation is from Udacity lesson "Jacobian Matrix Part 2".
MatrixXd Tools::CalculateJacobian(const VectorXd& x_state)
{
  // Calculate a Jacobian here.
  MatrixXd Hj(3,4);

  // Recover state parameters.
  float px = x_state(0);
  float py = x_state(1);
  float vx = x_state(2);
  float vy = x_state(3);

  // Pre-compute a set of terms to avoid repeated calculation.
  float c1 = px*px+py*py;
  float c2 = sqrt(c1);
  float c3 = (c1*c2);

  // Check division by zero.
  if (fabs(c1) < 0.0001)
  {
    throw std::runtime_error("CalculateJacobian() failed due to division by zero.");
  }

  // Compute the Jacobian matrix.
  Hj << (px / c2), (py / c2), 0, 0,
        -(py / c1), (px / c1), 0, 0,
        (py * (vx * py - vy * px) / c3), (px * (px * vy - py * vx) / c3), (px / c2), (py / c2);

  return Hj;
}

VectorXd Tools::ConvertCartesianToPolar(const VectorXd& x_state)
{
  // Recover state parameters.
  float px = x_state(0);
  float py = x_state(1);
  float vx = x_state(2);
  float vy = x_state(3);

  float c1 = sqrt(px*px+py*py);

  // Check division by zero.
  if (fabs(c1) < 0.0001)
  {
    throw std::runtime_error("CalculateJacobian() failed due to division by zero.");
  }

  VectorXd h(3, 1);
  h << c1,
       std::atan2(py, px),
       (px * vx + py * vy) / c1;

  return h;
}

VectorXd Tools::ConvertPolarToCartesian(const VectorXd& x)
{
  float rho = x(0);
  float phi = x(1);
  float rho_dot = x(2);

  float px = rho * std::cos(phi);
  float py = rho * std::sin(phi);
  float vx = 0;
  float vy = 0;

  VectorXd cartesian(4, 1);
  cartesian << px,
               py,
               vx,
               vy;

  return cartesian;
}
