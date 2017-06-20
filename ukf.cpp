#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 30;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 30;

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.0175;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.1;

  /**
  TODO:

  Complete the initialization. See ukf.h for other member properties.

  Hint: one or more values initialized above might be wildly off...
  */
  std_a_ = 2;
  std_yawdd_ = 0.5;
  
  x_(2) = 3;
  x_(3) = 0.0;
  x_(4) = 0.0;
  p_.fill(0.0);
  p_(0,0) = 1;
  p_(1,1) = 1;
  p_(2,2) = 1;
  p_(3,3) = 1;
  p_(4,4) = 1;
  
  n_x_ = 5;
  n_aug_ = 7;
  is_initialized_ = false;
  lambda_ = 3 - n_aug_;
  weights_ = VectorXd(2 * n_aug_ + 1);
}

UKF::~UKF() {}


void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /*****************************************************************************
   *  Initialization
   ****************************************************************************/
  if (!is_initialized_) {
    // first measurement
    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      /** Convert radar from polar to cartesian coordinates and initialize state. */
      double rho = measurement_pack.raw_measurements_[0];
      double phi = measurement_pack.raw_measurements_[1];
      double dot_rho = measurement_pack.raw_measurements_[2];
	  
      x_(0) = rho*cos(phi);
      x_(1) = rho*sin(phi);	  
    } else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
      /** Initialize state. */
      x_(0) = measurement_pack.raw_measurements_[0];
      x_(1) = measurement_pack.raw_measurements_[1];
    }
	
    //done initializing, no need to predict or update
    is_initialized_ = true;
    time_us_ = measurement_pack.timestamp_;
    return;
  }

 /*****************************************************************************
  *  Prediction
  ****************************************************************************/
  double delta_a = (measurement_pack.timestamp_ - time_us_) / 1000000;

  /* Step 1: Generate Sigma Points: already have x_ and P_ */
  MatrixXd Xsig_aug = MatrixxXd(n_aug_, 2 * n_aug_ + 1);
  AugmentedSigmaPoints(&Xsig_aug);
  /* Step 2: Sigma Point Prediction  */
  SigmaPointPrediction(&Xsig_pred_, &Xsig_aug, delta_t);
  /* Step 3: Predict Mean and Covariance Matrix  */
  VectorXd x_pred = VectorXd(n_x_); 
  MatrixXd P_pred = MatrixxXd(n_x_, n_x_);
  PredictMeanAndCovariance(&x_pred, &P_pred);
  /* Step 4: Predict Measurement*/
  VectorXd z_pred; 
  MatrixXd Zsig; 
  MatrixXd S_out; 

  if (measurement_pack.sensor_type :: MeasurementPackage::RADAR) {
      z_pred = VectorXd(3);
      S_out = MatrixXd(3, 2 * n_aug_ + 1);
      Zsig = MatrixXd(3, 2 * n_aug_ + 1);
      PredictMeasurementRadar(&z_pred, &S_out, &Zsig);
  } else if (measurement_pack.sensor_type :: MeasurementPackage::LASAR) {
      z_pred = VectorXd(2);
      S_out = MatrixXd(2, 2 * n_aug_ + 1);
      Zsig = MatrixXd(2, 2 * n_aug_ + 1);
      PredictMeasurementRadar(&z_pred, &S_out, &Zsig);
  } 

 /*****************************************************************************
  *  Prediction
  ****************************************************************************/
  VectorXd z_in;
  if (measurement_pack.sensor_type :: MeasurementPackage::RADAR) {
      z_in = VectotXd(3);
      z_in(0) = measurement_pack.raw_measurements_[0];
      z_in(1) = measurement_pack.raw_measurements_[1];
      z_in(2) = measurement_pack.raw_measurements_[2];
      UpdateStateRadar(&x_, &P_, &x_pred, &P_pred,
                       &Zsig, &z_pred, &S_out, &z_in);
  } else if (measurement_pack.sensor_type :: MeasurementPackage::LASAR) {
      z_in = VectotXd(3);
      z_in(0) = measurement_pack.raw_measurements_[0];
      z_in(1) = measurement_pack.raw_measurements_[1];
      z_in(2) = measurement_pack.raw_measurements_[2];
      UpdateStateLaser(&x_, &P_, &x_pred, &P_pred,
                       &Zsig, &z_pred, &S_out, &z_in);
  }
}

/* From x_ and P_ to get Xsig_aug */
void UKF::AugmentedSigmaPoints(MatrixXd* Xsig_out) {
  //set state dimension
  int n_x = n_x_;
  //set augmented dimension
  int n_aug = n_aug_;
  //Process noise standard deviation longitudinal acceleration in m/s^2
  double std_a = std_a_;
  //Process noise standard deviation yaw acceleration in rad/s^2
  double std_yawdd = std_yawdd_;
  //define spreading parameter
  double lambda = 3 - n_aug;

  //set example state
  VectorXd x = VectorXd(n_x);
  x = x_;
  //create example covariance matrix
  MatrixXd P = MatrixXd(n_x, n_x);
  P = P_;

  //create augmented mean vector
  VectorXd x_aug = VectorXd(n_aug);
  //create augmented state covariance
  MatrixXd P_aug = MatrixXd(n_aug, n_aug);

  //create sigma point matrix
  MatrixXd Xsig_aug = MatrixXd(n_aug, 2 * n_aug + 1);
 
  //create augmented mean state
  x_aug.head(n_x) = x;
  x_aug(5) = 0.0;
  x_aug(6) = 0.0;
  
  //create augmented covariance matrix
  P_aug.fill(0.0);
  P_aug.topLeftCorner(5,5) = P;
  P_aug(5,5) = std_a * std_a;
  P_aug(6,6) = std_yawdd * std_yawdd;
  
  //create square root matrix
  MatrixXd sqrt_p_aug = P_aug.llt().matrixL();
  
  //create augmented sigma points
  Xsig_aug.col(0) = x_aug;
  double temp = sqrt(lambda + n_aug);
  
  for (int i =1; i<=n_aug;i++) {
      Xsig_aug.col(i) = x_aug + temp*sqrt_p_aug.col(i-1);
      Xsig_aug.col(i+n_aug) = x_aug - temp*sqrt_p_aug.col(i-1);
  }
  
  //write result
  *Xsig_out = Xsig_aug;
}

void UKF::SigmaPointPrediction(MatrixXd* Xsig_pred_out,
                               MatrixXd* Xsig_in,
                               double delta_t)
{
  //set state dimension
  int n_x = n_x_;
  //set augmented dimension
  int n_aug = n_aug_;

  //create example sigma point matrix
  MatrixXd Xsig_aug = MatrixXd(n_aug, 2 * n_aug + 1);
  Xsig_aug = Xsig_in;
  
  //create matrix with predicted sigma points as columns
  MatrixXd Xsig_pred = MatrixXd(n_x, 2 * n_aug + 1);
 
  //predict sigma points
  for (int i = 0; i< 2*n_aug+1; i++)
  {
    //extract values for better readability
    double p_x = Xsig_aug(0,i);
    double p_y = Xsig_aug(1,i);
    double v = Xsig_aug(2,i);
    double yaw = Xsig_aug(3,i);
    double yawd = Xsig_aug(4,i);
    double nu_a = Xsig_aug(5,i);
    double nu_yawdd = Xsig_aug(6,i);

    //predicted state values
    double px_p, py_p;

    //avoid division by zero
    if (fabs(yawd) > 0.001) {
        px_p = p_x + v/yawd * ( sin (yaw + yawd*delta_t) - sin(yaw));
        py_p = p_y + v/yawd * ( cos(yaw) - cos(yaw+yawd*delta_t) );
    }
    else {
        px_p = p_x + v*delta_t*cos(yaw);
        py_p = p_y + v*delta_t*sin(yaw);
    }

    double v_p = v;
    double yaw_p = yaw + yawd*delta_t;
    double yawd_p = yawd;

    //add noise
    px_p = px_p + 0.5*nu_a*delta_t*delta_t * cos(yaw);
    py_p = py_p + 0.5*nu_a*delta_t*delta_t * sin(yaw);
    v_p = v_p + nu_a*delta_t;

    yaw_p = yaw_p + 0.5*nu_yawdd*delta_t*delta_t;
    yawd_p = yawd_p + nu_yawdd*delta_t;

    //write predicted sigma point into right column
    Xsig_pred(0,i) = px_p;
    Xsig_pred(1,i) = py_p;
    Xsig_pred(2,i) = v_p;
    Xsig_pred(3,i) = yaw_p;
    Xsig_pred(4,i) = yawd_p;
  }

  //write result
   *Xsig_pred_out = Xsig_pred;
}

void UKF::CalculateWeights()
{
  //set weights
  lambda_ = 3 - n_aug_;
  weights_(0) = lambda_ / (lambda_ + n_aug_);
  for (int i = 1; i < 2 * n_aug_ + 1; i++)
       weights_(i) = 0.5/(lambda_ + n_aug_);
}

void UKF::PredictMeanAndCovariance(VectorXd* x_out,
                                   MatrixXd* P_out)
{
  //set state dimension
  int n_x = n_x_;
  //set augmented dimension
  int n_aug = n_aug_;
  //define spreading parameter
  double lambda = 3 - n_aug;

  //create example matrix with predicted sigma points
  MatrixXd Xsig_pred = MatrixXd(n_x, 2 * n_aug + 1);
  Xsig_pred = Xsig_pred_;

  //create vector for predicted state
  VectorXd x = VectorXd(n_x);
  //create covariance matrix for prediction
  MatrixXd P = MatrixXd(n_x, n_x);

  //predict state mean
  x.fill(0.0);
  for (int i = 0; i < 2*n_aug+1; i++)
       x += weights_(i)*Xsig_pred.col(i);

  //predict state covariance matrix
  P.fill(0.0);
  for (int i = 0; i < 2*n_aug+1; i++) {
    VectorXd x_diff = Xsig_pred.col(i) - x;  
    
    while (x_diff(3) > M_PI) x_diff(3) -= (2 * M_PI);
    while (x_diff(3) < -M_PI) x_diff(3) += (2 * M_PI);
      
    P += weights_(i)*x_diff*x_diff.transpose();
  }
  
  //write result
  *x_out = x;
  *P_out = P;
}

void UKF::PredictMeasurementRadar(VectorXd* z_out,
                                  MatrixXd* S_out,
                                  MatrixXd* Zsig_out)
{

  //set state dimension
  int n_x = n_x_;
  //set augmented dimension
  int n_aug = n_aug_;
  //set measurement dimension, radar can measure r, phi, and r_dot
  int n_z = 3;
  //define spreading parameter
  double lambda = 3 - n_aug;


  //radar measurement noise standard deviation radius in m
  double std_radr = std_radr_;
  //radar measurement noise standard deviation angle in rad
  double std_radphi = std_radphi;
  //radar measurement noise standard deviation radius change in m/s
  double std_radrd = std_radrd_;

  //create example matrix with predicted sigma points
  MatrixXd Xsig_pred = MatrixXd(n_x, 2 * n_aug + 1);
  Xsig_pred = Xsig_pred_;

  //create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug + 1);
  //mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);
  //measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z,n_z);

  //transform sigma points into measurement space
  for (int i=0; i < 2*n_aug+1; i++) {
      double px = Xsig_pred(0,i);
      double py = Xsig_pred(1,i);
      double v = Xsig_pred(2,i);
      double yaw = Xsig_pred(3,i);
      double yaw_rate = Xsig_pred(4,i);
      
      double rho = sqrt(px*px + py*py);
      double phi = atan2(py,px);
      double rho_rate = (px*cos(yaw)*v + py*sin(yaw)*v) / rho;
      
      Zsig(0,i) = rho;
      Zsig(1,i) = phi;
      Zsig(2,i) = rho_rate;
  }
  
  //calculate mean predicted measurement
  z_pred.fill(0.0);
  for (int i=0; i < 2*n_aug+1; i++) {
      z_pred += weights_(i)*Zsig.col(i);
  }
  
  //calculate measurement covariance matrix S
  S.fill(0.0);
  for (int i=0; i <2*n_aug+1; i++) {
      VectorXd z_diff = Zsig.col(i) - z_pred;
      
      while (z_diff(1) > M_PI) z_diff(1) -= 2*M_PI;
      while (z_diff(1) < -M_PI) z_diff(1) += 2*M_PI;
      
      S += weights_(i)*z_diff*z_diff.transpose();
  }

  MatrixXd E = MatrixXd(n_z,n_z);
  E.fill(0.0);
  E(0,0) = std_radr*std_radr;
  E(1,1) = std_radphi*std_radphi;
  E(2,2) = std_radrd*std_radrd;
  S = S + E;
  
  //write result
  *z_out = z_pred;
  *S_out = S;
  *Zsig_out = Zsig;
}

void UKF::PredictRadarMeasurementLaser(VectorXd* z_out, MatrixXd* S_out) {
  //set state dimension
  int n_x = n_x_;
  //set augmented dimension
  int n_aug = n_aug_;
  //set measurement dimension, radar can measure r, phi, and r_dot
  int n_z = 2;

  //define spreading parameter
  double lambda = 3 - n_aug;

  //radar measurement noise standard deviation radius in m
  double std_px = std_laspx_;
  //radar measurement noise standard deviation angle in rad
  double std_py = std_laspy_;

  //create example matrix with predicted sigma points
  MatrixXd Xsig_pred = MatrixXd(n_x, 2 * n_aug + 1);
  Xsig_pred = Xsig_pred_;

  //create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug + 1);
  //mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);
  //measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z,n_z);

  //transform sigma points into measurement space
  for (int i=0; i < 2*n_aug+1; i++) {
      double px = Xsig_pred(0,i);
      double py = Xsig_pred(1,i);
      double v = Xsig_pred(2,i);
      double yaw = Xsig_pred(3,i);
      double yaw_rate = Xsig_pred(4,i);
      	 
      Zsig(0,i) = px;
      Zsig(1,i) = py;
  }
  
  //calculate mean predicted measurement
  z_pred.fill(0.0);
  for (int i=0; i < 2*n_aug+1; i++) {
      z_pred += weights_(i)*Zsig.col(i);
  }
  
  //calculate measurement covariance matrix S
  S.fill(0.0);
  for (int i=0; i <2*n_aug+1; i++) {
      VectorXd z_diff = Zsig.col(i) - z_pred;      
      S += weights_(i)*z_diff*z_diff.transpose();
  }

  MatrixXd E = MatrixXd(n_z,n_z);
  E.fill(0.0);
  E(0,0) = std_px*std_px;
  E(1,1) = std_py*std_py;
  S = S + E;
  
  //write result
  *z_out = z_pred;
  *S_out = S;
}

void UKF::UpdateStateRadar(VectorXd* x_out, MatrixXd* P_out,
                           VectorXd* x_pred, MatrixXd* P_pred,
                           MatrixXd* Zsig_in, VectorXd* z_pred_in,
                           MatrixXd* S_in, VectorXd* z_in)
{

  //set state dimension
  int n_x = n_x_;
  //set augmented dimension
  int n_aug = n_aug_;

  //set measurement dimension, radar can measure r, phi, and r_dot
  int n_z = 3;
  //define spreading parameter
  double lambda = 3 - n_aug;

  //create example matrix with predicted sigma points
  MatrixXd Xsig_pred = *Xsig_pred_;
  //create example vector for predicted state mean
  VectorXd x = *x_pred;
  //create example matrix for predicted state covariance
  MatrixXd P = *P_pred;
  MatrixXd Zsig = *Zsig_in;

  //create example vector for mean predicted measurement
  VectorXd z_pred = *z_pred_in;
  //create example matrix for predicted measurement covariance
  MatrixXd S = *S_in;  
  //create example vector for incoming radar measurement
  VectorXd z = *z_in;
  
  
  //create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x, n_z);

  //calculate cross correlation matrix
  Tc.fill(0.0);
  for (int i = 0; i < 2 * n_aug + 1; i++) {  //2n+1 simga points
    //residual
    VectorXd z_diff = Zsig.col(i) - z_pred;
    //angle normalization
    while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
    while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

    // state difference
    VectorXd x_diff = Xsig_pred.col(i) - x;
    //angle normalization
    while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
    while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;

    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }

  //Kalman gain K;
  MatrixXd K = Tc * S.inverse();

  //residual
  VectorXd z_diff = z - z_pred;

  //angle normalization
  while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
  while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

  //update state mean and covariance matrix
  x = x + K * z_diff;
  P = P - K*S*K.transpose();

  
  //write result
  *x_out = x;
  *P_out = P;
}

void UKF::UpdateStateLaser(VectorXd* x_out, MatrixXd* P_out,
                           VectorXd* x_pred, MatrixXd* P_pred,
                           MatrixXd* Zsig_in, VectorXd* z_pred_in,
                           MatrixXd* S_in, VectorXd* z_in)
{
  //set state dimension
  int n_x = n_x_;
  //set augmented dimension
  int n_aug = n_aug_;

  //set measurement dimension, radar can measure r, phi, and r_dot
  int n_z = 2;
  //define spreading parameter
  double lambda = 3 - n_aug;

 
  //create example matrix with predicted sigma points
  MatrixXd Xsig_pred = *Xsig_pred_;
  //create example vector for predicted state mean
  VectorXd x = *x_pred;
  //create example matrix for predicted state covariance
  MatrixXd P = *P_pred;
  MatrixXd Zsig = *Zsig_in;

  //create example vector for mean predicted measurement
  VectorXd z_pred = *z_pred_in;
  //create example matrix for predicted measurement covariance
  MatrixXd S = *S_in;  
  //create example vector for incoming radar measurement
  VectorXd z = *z_in;
  
  
  //create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x, n_z);

  //calculate cross correlation matrix
  Tc.fill(0.0);
  for (int i = 0; i < 2 * n_aug + 1; i++) {  //2n+1 simga points
    //residual
    VectorXd z_diff = Zsig.col(i) - z_pred;
    // state difference
    VectorXd x_diff = Xsig_pred.col(i) - x;
    //angle normalization
    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }

  //Kalman gain K;
  MatrixXd K = Tc * S.inverse();
  //residual
  VectorXd z_diff = z - z_pred;

  //angle normalization
  while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
  while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

  //update state mean and covariance matrix
  x = x + K * z_diff;
  P = P - K*S*K.transpose();
  
  //write result
  *x_out = x;
  *P_out = P;
}

void UKF::Prediction(double delta_t) {
  /**
  TODO:

  Complete this function! Estimate the object's location. Modify the state
  vector, x_. Predict sigma points, the state, and the state covariance matrix.
  */
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use lidar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the lidar NIS.
  */
  
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use radar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the radar NIS.
  */
}
