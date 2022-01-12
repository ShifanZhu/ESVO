#include <esvo_time_surface/TimeSurface.h>
#include <esvo_time_surface/TicToc.h>
#include <opencv2/calib3d/calib3d.hpp>
#include <std_msgs/Float32.h>
#include <glog/logging.h>
#include <thread>

//#define ESVO_TS_LOG

namespace esvo_time_surface 
{
TimeSurface::TimeSurface(ros::NodeHandle & nh, ros::NodeHandle nh_private)
  : nh_(nh)
{
  T_W_I_ = Eigen::Matrix4d::Identity();
  // setup subscribers and publishers
  event_sub_ = nh_.subscribe("/dvs/events", 0, &TimeSurface::eventsCallback, this);
  camera_info_sub_ = nh_.subscribe("/dvs/camera_info", 1, &TimeSurface::cameraInfoCallback, this);
  sync_topic_ = nh_.subscribe("sync", 1, &TimeSurface::syncCallback, this);
  imu_sub_ = nh_.subscribe("/dvs/imu", 1, &TimeSurface::imuCallback, this);
  image_transport::ImageTransport it_(nh_);
  time_surface_pub_ = it_.advertise("time_surface", 1);

  // parameters
  nh_private.param<bool>("use_sim_time", bUse_Sim_Time_, true);
  nh_private.param<bool>("ignore_polarity", ignore_polarity_, true);
  nh_private.param<bool>("time_surface_at_most_recent_event", time_surface_at_most_recent_event_, true);
  nh_private.param<double>("decay_ms", decay_ms_, 30);
  int TS_mode;
  nh_private.param<int>("time_surface_mode", TS_mode, 0);
  time_surface_mode_ = (TimeSurfaceMode)TS_mode;
  nh_private.param<int>("median_blur_kernel_size", median_blur_kernel_size_, 1);
  nh_private.param<int>("max_event_queue_len", max_event_queue_length_, 20);
  MAX_EVENT_QUEUE_LENGTH = 30000;
  //
  bCamInfoAvailable_ = false;
  bSensorInitialized_ = false;
  if(pEventQueueMat_)
    pEventQueueMat_->clear();
  sensor_size_ = cv::Size(0,0);

  init_time=clock();
  current_time=clock();
  g_last_imu_time = -1.0;
  imu_inited_ = false;
  imu_cnt_ = 0.0;
  tmp_P=Eigen::Vector3d(0, 0, 0); //t
  tmp_Q=Eigen::Quaterniond::Identity();//R
  tmp_V=Eigen::Vector3d(0, 0, 0);
  vel_W=Eigen::Vector3d(0, 0, 0);
  acc_bias_ = Eigen::Vector3d::Zero();
  gyr_bias_ = Eigen::Vector3d::Zero();
  g_ = Eigen::Vector3d(0., 0., 9.81);
  g_imu_path_pub = nh_.advertise<nav_msgs::Path>("imu_path",1, true);
  g_imu_path.header.frame_id="map";
	localizationPosePub_ = nh_.advertise<geometry_msgs::PoseStamped>("imu_pose", 1);
  projection_mode_ = 2;
  combine_frame_size_ = 2;
}

TimeSurface::~TimeSurface()
{
  time_surface_pub_.shutdown();
}

void TimeSurface::init(int width, int height)
{
  sensor_size_ = cv::Size(width, height);
  bSensorInitialized_ = true;
  pEventQueueMat_.reset(new EventQueueMat(width, height, max_event_queue_length_));
  ROS_INFO("Sensor size: (%d x %d)", sensor_size_.width, sensor_size_.height);



  // TODO FIXED NUMBER !!!!!!!!!
  bCamInfoAvailable_ = true;
  camera_matrix_ = cv::Mat::zeros(3, 3, CV_64F);
  // std::cout << " camera_matrix_" << std::endl << camera_matrix_ << std::endl;
  camera_matrix_.at<double>(0, 0) = 246.80452042602624;
  camera_matrix_.at<double>(1, 1) = 247.2360195819396;
  camera_matrix_.at<double>(0, 2) = 175.11132754932393;
  camera_matrix_.at<double>(1, 2) = 117.89262595334824;
  // std::cout << " camera_matrix_" << std::endl << camera_matrix_ << std::endl;
  distortion_params_ << -0.3639965489793874, 0.1250383440346353, -0.0009396955074747349, -0.0012786111121125062, 0.0;
  projection_params_ << camera_matrix_.at<double>(0, 0),camera_matrix_.at<double>(1, 1),camera_matrix_.at<double>(0, 2),camera_matrix_.at<double>(1, 2);
  calculateBearingLUT(&(dvs_bearing_lut_));
  calculateKeypointLUT(dvs_bearing_lut_, &dvs_keypoint_lut_);

}

// template
void TimeSurface::distort(const double* params, double* px, double* jac_colmajor /*= nullptr*/)
{
  const double x = px[0];
  const double y = px[1];
  const double k1 = params[0];
  const double k2 = params[1];
  const double k3 = params[2];
  const double k4 = params[3];
  const double r_sqr = x * x + y * y;
  const double r = std::sqrt(r_sqr);
  const double theta = std::atan(r);
  const double theta2 = theta * theta;
  const double theta4 = theta2 * theta2;
  const double theta6 = theta4 * theta2;
  const double theta8 = theta4 * theta4;
  const double thetad = theta * (1.0 + k1 * theta2 + k2 * theta4 + k3 * theta6 + k4 * theta8);
  const double scaling = (r > 1e-8) ? thetad / r : 1.0;
  px[0] *= scaling;
  px[1] *= scaling;

  if (jac_colmajor)
  {
    double& J_00 = jac_colmajor[0];
    double& J_10 = jac_colmajor[1];
    double& J_01 = jac_colmajor[2];
    double& J_11 = jac_colmajor[3];

    if(r < 1e-7)
    {
      J_00 = 1.0; J_01 = 0.0;
      J_10 = 0.0; J_11 = 1.0;
    }
    else
    {
      double xx = x * x;
      double yy = y * y;
      double xy = x * y;
      double theta_inv_r = theta / r;
      double theta_sqr = theta * theta;
      double theta_four = theta_sqr * theta_sqr;

      double t1 = 1.0 / (xx + yy + 1.0);
      double t2 = k1 * theta_sqr
            + k2 * theta_four
            + k3 * theta_four * theta_sqr
            + k4 * (theta_four * theta_four) + 1.0;
      double t3 = t1 * theta_inv_r;

      double offset = t2 * theta_inv_r;
      double scale  = t2 * (t1 / r_sqr - theta_inv_r / r_sqr)
          + theta_inv_r * t3 * (
                2.0 * k1
              + 4.0 * k2 * theta_sqr
              + 6.0 * k3 * theta_four
              + 8.0 * k4 * theta_four * theta_sqr);

      J_11 = yy * scale + offset;
      J_00 = xx * scale + offset;
      J_01 = xy * scale;
      J_10 = J_01;
    }
  }
}

void TimeSurface::undistort(const double* params, double* px)
{
  double jac_colmajor[4];
  double x[2];
  double x_tmp[2];
  x[0] = px[0]; x[1]= px[1];
  for(int i = 0; i < 30; ++i)
  {
    x_tmp[0] = x[0]; x_tmp[1] = x[1];
    distort(params, x_tmp, jac_colmajor);

    const double e_u = px[0] - x_tmp[0];
    const double e_v = px[1] - x_tmp[1];

    const double a = jac_colmajor[0];
    const double b = jac_colmajor[1];
    const double d = jac_colmajor[3];

    // direct gauss newton step
    const double a_sqr = a * a;
    const double b_sqr = b * b;
    const double d_sqr = d * d;
    const double abbd = a * b + b * d;
    const double abbd_sqr = abbd * abbd;
    const double a2b2 = a_sqr + b_sqr;
    const double a2b2_inv = 1.0 / a2b2;
    const double adabdb = a_sqr * d_sqr - 2 * a * b_sqr * d + b_sqr * b_sqr;
    const double adabdb_inv = 1.0 / adabdb;
    const double c1 = abbd * adabdb_inv;

    x[0] += e_u * (a * (abbd_sqr * a2b2_inv * adabdb_inv + a2b2_inv) - b * c1) + e_v * (b * (abbd_sqr * a2b2_inv * adabdb_inv + a2b2_inv) - d * c1);
    x[1] += e_u * (-a * c1 + b * a2b2 * adabdb_inv) + e_v * (-b * c1 + d * a2b2 * adabdb_inv);

    if ((e_u * e_u + e_v * e_v) < 1e-8)
    {
      break;
    }
  }

  px[0] = x[0];
  px[1] = x[1];
}

Keypoint TimeSurface::project(const Eigen::Ref<const Bearing>& bearing)
{
  // Unit coordinates -> distortion -> pinhole, offset and scale.
  Keypoint px = bearing.head<2>() / bearing(2);
  distort(this->distortion_params_.data(), px.data());
  project(this->projection_params_.data(), px.data());
  return px;
}

void TimeSurface::project(const double* params, double* px)
{
  const double fx = params[0];
  const double fy = params[1];
  const double cx = params[2];
  const double cy = params[3];
  px[0] = px[0] * fx + cx;
  px[1] = px[1] * fy + cy;
}

Bearing TimeSurface::backProject(const Eigen::Ref<const Keypoint>& px)
{
  Bearing bearing;
  bearing << px(0), px(1), 1.0;
  backProject(this->projection_params_.data(), bearing.data());
  undistort(this->distortion_params_.data(), bearing.data());
  return bearing.normalized();
}

// double -> T
void TimeSurface::backProject(const double* params, double* px)
{
  const double fx = params[0];
  const double fy = params[1];
  const double cx = params[2];
  const double cy = params[3];
  px[0] = (px[0] - cx) / fx;
  px[1] = (px[1] - cy) / fy;
}

void TimeSurface::calculateBearingLUT(Eigen::Matrix<double, 4, Eigen::Dynamic>* dvs_bearing_lut)
{
  // CHECK_NOTNULL(dvs_bearing_lut);
  size_t n = sensor_size_.height * sensor_size_.width;
  dvs_bearing_lut->resize(4, n);

  for (size_t y=0; y != sensor_size_.height; ++y)
  {
    for (size_t x=0; x != sensor_size_.width; ++x)
    {
      // This back projects keypoints and undistorts the
      // image into bearing vectors.
      Bearing f = backProject(Keypoint(x,y));
      dvs_bearing_lut->col(x + y * sensor_size_.width) =
          Eigen::Vector4d(f[0], f[1], f[2], 1.);
    }
  }
}

void TimeSurface::calculateKeypointLUT(const Eigen::Matrix<double, 4, Eigen::Dynamic>& dvs_bearing_lut,
                                    Eigen::Matrix<double, 2, Eigen::Dynamic>* dvs_keypoint_lut)
{
  // CHECK_NOTNULL(dvs_keypoint_lut);
  size_t n = sensor_size_.height * sensor_size_.width;
  // CHECK(n == static_cast<size_t>(dvs_bearing_lut.cols())) << "Size of bearing"
  //                                                             " lut is not consistent with camera.";

  dvs_keypoint_lut->resize(2, n);
  for (size_t i=0; i != n; ++i)
  {
    Keypoint p = project(
          dvs_bearing_lut.col(i).head<3>().cast<double>());
    dvs_keypoint_lut->col(i) = p.cast<double>();
  }
}

void TimeSurface::createTimeSurfaceAtTime(const ros::Time& external_sync_time)
{
  std::lock_guard<std::mutex> lock(data_mutex_);

  if(!bSensorInitialized_ || !bCamInfoAvailable_)
    return;

  // create exponential-decayed Time Surface map.
  const double decay_sec = decay_ms_ / 1000.0;
  cv::Mat time_surface_map;
  time_surface_map = cv::Mat::zeros(sensor_size_, CV_64F);

  // Loop through all coordinates
  for(int y=0; y<sensor_size_.height; ++y)
  {
    for(int x=0; x<sensor_size_.width; ++x)
    {
      dvs_msgs::Event most_recent_event_at_coordXY_before_T;
      if(pEventQueueMat_->getMostRecentEventBeforeT(x, y, external_sync_time, &most_recent_event_at_coordXY_before_T))
      {
        const ros::Time& most_recent_stamp_at_coordXY = most_recent_event_at_coordXY_before_T.ts;
        if(most_recent_stamp_at_coordXY.toSec() > 0)
        {
          // Get delta time: specified timestamp minus most recent timestamp
          const double dt = (external_sync_time - most_recent_stamp_at_coordXY).toSec();
          double polarity = (most_recent_event_at_coordXY_before_T.polarity) ? 1.0 : -1.0;
          double expVal = std::exp(-dt / decay_sec);
          if(!ignore_polarity_)
            expVal *= polarity;

          // Time Surface Mode
          // Backward: First Apply exp decay on the raw image plane, then get the value
          //           at each pixel in the rectified image plane by looking up the
          //           corresponding one (float coordinates) with bi-linear interpolation.
          // Forward: First warp the raw events to the rectified image plane, then
          //          apply the exp decay on the four neighbouring (involved) pixel coordinate.

          // Backward version --> Directly use exp decay value
          if(time_surface_mode_ == BACKWARD)
            time_surface_map.at<double>(y,x) = expVal;

          // Forward version
          if(time_surface_mode_ == FORWARD && bCamInfoAvailable_)
          {
            /* pre-compute the undistorted-rectified look-up table, this table undistrots and maps point 
              from x*y coord to a 1*xy table */
            Eigen::Matrix<double, 2, 1> uv_rect = precomputed_rectified_points_.block<2, 1>(0, y * sensor_size_.width + x);
            size_t u_i, v_i;
            if(uv_rect(0) >= 0 && uv_rect(1) >= 0)
            {
              u_i = std::floor(uv_rect(0));
              v_i = std::floor(uv_rect(1));

              // Four neighbouring (involved) pixel coordinate
              if(u_i + 1 < sensor_size_.width && v_i + 1 < sensor_size_.height)
              {
                double fu = uv_rect(0) - u_i; // Get fractional part of uv_rect(0)
                double fv = uv_rect(1) - v_i;
                double fu1 = 1.0 - fu; // Get fractional part of uv_rect(0) to 1
                double fv1 = 1.0 - fv;
                time_surface_map.at<double>(v_i, u_i) += fu1 * fv1 * expVal;
                time_surface_map.at<double>(v_i, u_i + 1) += fu * fv1 * expVal;
                time_surface_map.at<double>(v_i + 1, u_i) += fu1 * fv * expVal;
                time_surface_map.at<double>(v_i + 1, u_i + 1) += fu * fv * expVal;

                if(time_surface_map.at<double>(v_i, u_i) > 1)
                  time_surface_map.at<double>(v_i, u_i) = 1;
                if(time_surface_map.at<double>(v_i, u_i + 1) > 1)
                  time_surface_map.at<double>(v_i, u_i + 1) = 1;
                if(time_surface_map.at<double>(v_i + 1, u_i) > 1)
                  time_surface_map.at<double>(v_i + 1, u_i) = 1;
                if(time_surface_map.at<double>(v_i + 1, u_i + 1) > 1)
                  time_surface_map.at<double>(v_i + 1, u_i + 1) = 1;
              }
            }
          } // forward
        }
      } // a most recent event is available
    }// loop x
  }// loop y

  // polarity
  if(!ignore_polarity_)
    time_surface_map = 255.0 * (time_surface_map + 1.0) / 2.0;
  else
    time_surface_map = 255.0 * time_surface_map;
  time_surface_map.convertTo(time_surface_map, CV_8U);

  // median blur
  if(median_blur_kernel_size_ > 0)
    cv::medianBlur(time_surface_map, time_surface_map, 2 * median_blur_kernel_size_ + 1);

  // Publish event image
  static cv_bridge::CvImage cv_image;
  cv_image.encoding = "mono8";
  cv_image.image = time_surface_map.clone();

  if(time_surface_mode_ == FORWARD && time_surface_pub_.getNumSubscribers() > 0)
  {
    cv_image.header.stamp = external_sync_time;
    time_surface_pub_.publish(cv_image.toImageMsg());
  }

  if (time_surface_mode_ == BACKWARD && bCamInfoAvailable_ && time_surface_pub_.getNumSubscribers() > 0)
  {
    cv_bridge::CvImage cv_image2;
    cv_image2.encoding = cv_image.encoding;
    cv_image2.header.stamp = external_sync_time;
    cv::remap(cv_image.image, cv_image2.image, undistort_map1_, undistort_map2_, CV_INTER_LINEAR);
    time_surface_pub_.publish(cv_image2.toImageMsg());
  }
}

void TimeSurface::createTimeSurfaceAtMostRecentEvent()
{
  std::lock_guard<std::mutex> lock(data_mutex_);

  if(!bSensorInitialized_ || !bCamInfoAvailable_)
    return;
  // create exponential-decayed Time Surface map.
  const double decay_sec = decay_ms_ / 1000.0;
  cv::Mat time_surface_map;
  time_surface_map = cv::Mat::zeros(sensor_size_, CV_64F);

  // EventQueue& event_00 = getEventQueue(0, 0);
  // auto it = eq.rbegin();
  // const dvs_msgs::Event& e = *it;

  ros::Time time_now_m1 = ros::Time::now()-ros::Duration(1);
  // std::cout << "time_now_m1" << time_now_m1 << std::endl;
  // Loop through all coordinates
  for(int y=0; y<sensor_size_.height; ++y)
  {
    for(int x=0; x<sensor_size_.width; ++x)
    {
      // No event at xy: false
      EventQueue& eq = pEventQueueMat_->getEventQueue(x, y);
      if(eq.empty())
        continue;

      // Loop through all events to find most recent event
      // Assume events are ordered from latest to oldest
      // for(auto it = eq.rbegin(); it != eq.rend(); ++it)
      // {
        auto it = eq.rbegin();
        const dvs_msgs::Event& e = *it;
        // auto time_temp = e.ts;
        if(e.ts > time_now_m1)
        {
          time_now_m1 = e.ts;
        }
      // }
    }
  }
  ros::Time time_now_m2 = time_now_m1 + ros::Duration(0.02);

  int cnt = 0;

  // Loop through all coordinates
  for(int y=0; y<sensor_size_.height; ++y)
  {
    for(int x=0; x<sensor_size_.width; ++x)
    {
      dvs_msgs::Event most_recent_event_at_coordXY_before_T;
      if(pEventQueueMat_->getMostRecentEventBeforeT(x, y, time_now_m2, &most_recent_event_at_coordXY_before_T))
      {
        const ros::Time& most_recent_stamp_at_coordXY = most_recent_event_at_coordXY_before_T.ts;
        if(most_recent_stamp_at_coordXY.toSec() > 0)
        {
          // Get delta time: specified timestamp minus most recent timestamp
          const double dt = (time_now_m1 - most_recent_stamp_at_coordXY).toSec();
          double expVal = 0;  
          if(dt > 0.01) continue;
          // std::cout << "dt == " << dt << std::endl;
          else if (dt < 0.01)
          {
            expVal = 1;//std::exp(-0.3*dt / decay_sec);
          }
          else
          {
            expVal = 1;//std::exp(-0.3*dt / decay_sec);
          }
          
          // double expVal = (dt / decay_sec);
          double polarity = (most_recent_event_at_coordXY_before_T.polarity) ? 1.0 : -1.0;
          if(!ignore_polarity_)
            expVal *= polarity;

          // Time Surface Mode
          // Backward: First Apply exp decay on the raw image plane, then get the value
          //           at each pixel in the rectified image plane by looking up the
          //           corresponding one (float coordinates) with bi-linear interpolation.
          // Forward: First warp the raw events to the rectified image plane, then
          //          apply the exp decay on the four neighbouring (involved) pixel coordinate.

          // Backward version --> Directly use exp decay value
          if(time_surface_mode_ == BACKWARD && x<sensor_size_.width && y<sensor_size_.height)
            time_surface_map.at<double>(y,x) = expVal;

          // Forward version
          if(time_surface_mode_ == FORWARD && bCamInfoAvailable_)
          {
            /* pre-compute the undistorted-rectified look-up table, this table undistrots and maps point 
              from x*y coord to a 1*xy table */
            Eigen::Matrix<double, 2, 1> uv_rect = precomputed_rectified_points_.block<2, 1>(0, y * sensor_size_.width + x);
            size_t u_i, v_i;
            if(uv_rect(0) >= 0 && uv_rect(1) >= 0)
            {
              u_i = std::floor(uv_rect(0));
              v_i = std::floor(uv_rect(1));

              // Four neighbouring (involved) pixel coordinate
              if(u_i + 1 < sensor_size_.width && v_i + 1 < sensor_size_.height)
              {
                double fu = uv_rect(0) - u_i; // Get fractional part of uv_rect(0)
                double fv = uv_rect(1) - v_i;
                double fu1 = 1.0 - fu; // Get fractional part of uv_rect(0) to 1
                double fv1 = 1.0 - fv;
                time_surface_map.at<double>(v_i, u_i) += fu1 * fv1 * expVal;
                time_surface_map.at<double>(v_i, u_i + 1) += fu * fv1 * expVal;
                time_surface_map.at<double>(v_i + 1, u_i) += fu1 * fv * expVal;
                time_surface_map.at<double>(v_i + 1, u_i + 1) += fu * fv * expVal;

                // if(time_surface_map.at<double>(v_i, u_i) > 0.9)
                // {
                //   cnt++;
                //   double ratio = (double) cnt / (double) sensor_size_.height / (double) sensor_size_.width;
                //   std::cout << "ratio & time surface map = " << ratio << "   " << time_surface_map.at<double>(v_i, u_i) << std::endl;
                // }

                if(time_surface_map.at<double>(v_i, u_i) > 1)
                {
                  time_surface_map.at<double>(v_i, u_i) = 1;
                }  
                if(time_surface_map.at<double>(v_i, u_i + 1) > 1)
                {

                  time_surface_map.at<double>(v_i, u_i + 1) = 1;
                }
                if(time_surface_map.at<double>(v_i + 1, u_i) > 1)
                {
                  time_surface_map.at<double>(v_i + 1, u_i) = 1;
                }
                if(time_surface_map.at<double>(v_i + 1, u_i + 1) > 1)
                {
                  time_surface_map.at<double>(v_i + 1, u_i + 1) = 1;
                }
              }
            }
          } // forward
        }
      } // a most recent event is available
    }// loop x
  }// loop y

  // polarity
  if(!ignore_polarity_)
    time_surface_map = 255.0 * (time_surface_map + 1.0) / 2.0;
  else
    // time_surface_map = 255.0 * time_surface_map;
    time_surface_map = 255.0 * (-time_surface_map+1);
  time_surface_map.convertTo(time_surface_map, CV_8U);

  // median blur
  if(median_blur_kernel_size_ > 0)
    cv::medianBlur(time_surface_map, time_surface_map, 2 * median_blur_kernel_size_ + 1);

  // Publish event image
  static cv_bridge::CvImage cv_image;
  cv_image.encoding = "mono8";
  cv_image.image = time_surface_map.clone();

  if(time_surface_mode_ == FORWARD && time_surface_pub_.getNumSubscribers() > 0)
  {
    cv_image.header.stamp = time_now_m1;
    time_surface_pub_.publish(cv_image.toImageMsg());
  }

  if (time_surface_mode_ == BACKWARD && bCamInfoAvailable_ && time_surface_pub_.getNumSubscribers() > 0)
  {
    cv_bridge::CvImage cv_image2;
    cv_image2.encoding = cv_image.encoding;
    cv_image2.header.stamp = time_now_m1;
    cv::remap(cv_image.image, cv_image2.image, undistort_map1_, undistort_map2_, CV_INTER_LINEAR);
    time_surface_pub_.publish(cv_image2.toImageMsg());
  }
}

void TimeSurface::createTimeSurfaceAtTime_hyperthread(const ros::Time& external_sync_time)
{
  std::lock_guard<std::mutex> lock(data_mutex_);

  if(!bSensorInitialized_ || !bCamInfoAvailable_)
    return;

  // create exponential-decayed Time Surface map.
  const double decay_sec = decay_ms_ / 1000.0;
  cv::Mat time_surface_map;
  time_surface_map = cv::Mat::zeros(sensor_size_, CV_64F);

  // distribute jobs
  std::vector<Job> jobs(NUM_THREAD_TS);
  size_t num_col_per_thread = sensor_size_.width / NUM_THREAD_TS;
  size_t res_col = sensor_size_.width % NUM_THREAD_TS;
  for(size_t i = 0; i < NUM_THREAD_TS; i++)
  {
    jobs[i].i_thread_ = i;
    jobs[i].pEventQueueMat_ = pEventQueueMat_.get();
    jobs[i].pTimeSurface_ = &time_surface_map;
    jobs[i].start_col_ = num_col_per_thread * i;
    if(i == NUM_THREAD_TS - 1)
      jobs[i].end_col_ = jobs[i].start_col_ + num_col_per_thread - 1 + res_col;
    else
      jobs[i].end_col_ = jobs[i].start_col_ + num_col_per_thread - 1;
    jobs[i].start_row_ = 0;
    jobs[i].end_row_ = sensor_size_.height - 1;
    jobs[i].external_sync_time_ = external_sync_time;
    jobs[i].decay_sec_ = decay_sec;
  }

  // hyper thread processing
  std::vector<std::thread> threads;
  threads.reserve(NUM_THREAD_TS);
  for(size_t i = 0; i < NUM_THREAD_TS; i++)
    threads.emplace_back(std::bind(&TimeSurface::thread, this, jobs[i]));
  for(auto& thread:threads)
    if(thread.joinable())
      thread.join();

  // polarity
  if(!ignore_polarity_)
    time_surface_map = 255.0 * (time_surface_map + 1.0) / 2.0;
  else
    time_surface_map = 255.0 * time_surface_map;
  time_surface_map.convertTo(time_surface_map, CV_8U);

  // median blur
  if(median_blur_kernel_size_ > 0)
    cv::medianBlur(time_surface_map, time_surface_map, 2 * median_blur_kernel_size_ + 1);

  // Publish event image
  static cv_bridge::CvImage cv_image;
  cv_image.encoding = "mono8";
  cv_image.image = time_surface_map.clone();

  if(time_surface_mode_ == FORWARD && time_surface_pub_.getNumSubscribers() > 0)
  {
    cv_image.header.stamp = external_sync_time;
    time_surface_pub_.publish(cv_image.toImageMsg());
  }

  if (time_surface_mode_ == BACKWARD && bCamInfoAvailable_ && time_surface_pub_.getNumSubscribers() > 0)
  {
    cv_bridge::CvImage cv_image2;
    cv_image2.encoding = cv_image.encoding;
    cv_image2.header.stamp = external_sync_time;
    cv::remap(cv_image.image, cv_image2.image, undistort_map1_, undistort_map2_, CV_INTER_LINEAR);
    time_surface_pub_.publish(cv_image2.toImageMsg());
  }
}

void TimeSurface::thread(Job &job)
{
  EventQueueMat & eqMat = *job.pEventQueueMat_;
  cv::Mat& time_surface_map = *job.pTimeSurface_;
  size_t start_col = job.start_col_;
  size_t end_col = job.end_col_;
  size_t start_row = job.start_row_;
  size_t end_row = job.end_row_;
  size_t i_thread = job.i_thread_;

  for(size_t y = start_row; y <= end_row; y++)
    for(size_t x = start_col; x <= end_col; x++)
    {
      dvs_msgs::Event most_recent_event_at_coordXY_before_T;
      if(pEventQueueMat_->getMostRecentEventBeforeT(x, y, job.external_sync_time_, &most_recent_event_at_coordXY_before_T))
      {
        const ros::Time& most_recent_stamp_at_coordXY = most_recent_event_at_coordXY_before_T.ts;
        if(most_recent_stamp_at_coordXY.toSec() > 0)
        {
          const double dt = (job.external_sync_time_ - most_recent_stamp_at_coordXY).toSec();
          double polarity = (most_recent_event_at_coordXY_before_T.polarity) ? 1.0 : -1.0;
          double expVal = std::exp(-dt / job.decay_sec_);
          if(!ignore_polarity_)
            expVal *= polarity;

          // Backward version
          if(time_surface_mode_ == BACKWARD)
            time_surface_map.at<double>(y,x) = expVal;

          // Forward version
          if(time_surface_mode_ == FORWARD && bCamInfoAvailable_)
          {
            Eigen::Matrix<double, 2, 1> uv_rect = precomputed_rectified_points_.block<2, 1>(0, y * sensor_size_.width + x);
            size_t u_i, v_i;
            if(uv_rect(0) >= 0 && uv_rect(1) >= 0)
            {
              u_i = std::floor(uv_rect(0));
              v_i = std::floor(uv_rect(1));

              if(u_i + 1 < sensor_size_.width && v_i + 1 < sensor_size_.height)
              {
                double fu = uv_rect(0) - u_i;
                double fv = uv_rect(1) - v_i;
                double fu1 = 1.0 - fu;
                double fv1 = 1.0 - fv;
                time_surface_map.at<double>(v_i, u_i) += fu1 * fv1 * expVal;
                time_surface_map.at<double>(v_i, u_i + 1) += fu * fv1 * expVal;
                time_surface_map.at<double>(v_i + 1, u_i) += fu1 * fv * expVal;
                time_surface_map.at<double>(v_i + 1, u_i + 1) += fu * fv * expVal;

                if(time_surface_map.at<double>(v_i, u_i) > 1)
                  time_surface_map.at<double>(v_i, u_i) = 1;
                if(time_surface_map.at<double>(v_i, u_i + 1) > 1)
                  time_surface_map.at<double>(v_i, u_i + 1) = 1;
                if(time_surface_map.at<double>(v_i + 1, u_i) > 1)
                  time_surface_map.at<double>(v_i + 1, u_i) = 1;
                if(time_surface_map.at<double>(v_i + 1, u_i + 1) > 1)
                  time_surface_map.at<double>(v_i + 1, u_i + 1) = 1;
              }
            }
          } // forward
        }
      } // a most recent event is available
    }
}

void TimeSurface::syncCallback(const std_msgs::TimeConstPtr& msg)
{
  if(bUse_Sim_Time_)
    sync_time_ = ros::Time::now();
  else
    sync_time_ = msg->data;

#ifdef ESVO_TS_LOG
    TicToc tt;
    tt.tic();
#endif
    if(NUM_THREAD_TS == 1 && !time_surface_at_most_recent_event_)
      createTimeSurfaceAtTime(sync_time_);
    if(NUM_THREAD_TS > 1)
      createTimeSurfaceAtTime_hyperthread(sync_time_);
    if(NUM_THREAD_TS == 1 && time_surface_at_most_recent_event_)
      createTimeSurfaceAtMostRecentEvent();
#ifdef ESVO_TS_LOG
    LOG(INFO) << "Time Surface map's creation takes: " << tt.toc() << " ms.";
#endif
}

void TimeSurface::cameraInfoCallback(const sensor_msgs::CameraInfo::ConstPtr& msg)
{
  if(bCamInfoAvailable_)
    return;

  sensor_size_ = cv::Size(msg->width, msg->height);


  // cv::Size sensor_size(msg->width, msg->height);
  camera_matrix_ = cv::Mat(3, 3, CV_64F);
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++)
      camera_matrix_.at<double>(cv::Point(i, j)) = msg->K[i+j*3];

  distortion_model_ = msg->distortion_model;
  dist_coeffs_ = cv::Mat(msg->D.size(), 1, CV_64F);
  for (int i = 0; i < msg->D.size(); i++)
    dist_coeffs_.at<double>(i) = msg->D[i];

  rectification_matrix_ = cv::Mat(3, 3, CV_64F);
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++)
      rectification_matrix_.at<double>(cv::Point(i, j)) = msg->R[i+j*3];

  projection_matrix_ = cv::Mat(3, 4, CV_64F);
  for (int i = 0; i < 4; i++)
    for (int j = 0; j < 3; j++)
      projection_matrix_.at<double>(cv::Point(i, j)) = msg->P[i+j*4];

  if(distortion_model_ == "equidistant")
  {
    // cv::fisheye::initUndistortRectifyMap(camera_matrix_, dist_coeffs_,
    //                                      rectification_matrix_, projection_matrix_,
    //                                      sensor_size_, CV_32FC1, undistort_map1_, undistort_map2_);
    bCamInfoAvailable_ = true;
    ROS_INFO("Camera information is loaded (Distortion model %s).", distortion_model_.c_str());
  }
  else if(distortion_model_ == "plumb_bob")
  {
    // cv::initUndistortRectifyMap(camera_matrix_, dist_coeffs_,
    //                             rectification_matrix_, projection_matrix_,
    //                             sensor_size_, CV_32FC1, undistort_map1_, undistort_map2_);
    bCamInfoAvailable_ = true;
    ROS_INFO("Camera information is loaded (Distortion model %s).", distortion_model_.c_str());
  }
  else
  {
    ROS_ERROR_ONCE("Distortion model %s is not supported.", distortion_model_.c_str());
    bCamInfoAvailable_ = false;
    return;
  }

  /* pre-compute the undistorted-rectified look-up table */
  precomputed_rectified_points_ = Eigen::Matrix2Xd(2, sensor_size_.height * sensor_size_.width);
  // raw coordinates
  cv::Mat_<cv::Point2f> RawCoordinates(1, sensor_size_.height * sensor_size_.width);
  for (int y = 0; y < sensor_size_.height; y++)
  {
    for (int x = 0; x < sensor_size_.width; x++)
    {
      int index = y * sensor_size_.width + x;
      RawCoordinates(index) = cv::Point2f((double) x, (double) y);
    }
  }
  // undistorted-rectified coordinates
  cv::Mat_<cv::Point2f> RectCoordinates(1, sensor_size_.height * sensor_size_.width);
  for(int i = 0; i < sensor_size_.height * sensor_size_.width; ++i)
  {
    RectCoordinates(i) = RawCoordinates(i);
  }
  std::cout << "direct apply points!! Do NOT do undistrotion!" << std::endl;
  if (distortion_model_ == "plumb_bob")
  {
    // TODO: Take care that the undistort function is commented !!!
    // cv::undistortPoints(RawCoordinates, RectCoordinates, camera_matrix_, dist_coeffs_,
    //                     rectification_matrix_, projection_matrix_);
    ROS_INFO("Undistorted-Rectified Look-Up Table with Distortion model: %s", distortion_model_.c_str());
  }
  else if (distortion_model_ == "equidistant")
  {
    // cv::fisheye::undistortPoints(
    //   RawCoordinates, RectCoordinates, camera_matrix_, dist_coeffs_,
    //   rectification_matrix_, projection_matrix_);
    ROS_INFO("Undistorted-Rectified Look-Up Table with Distortion model: %s", distortion_model_.c_str());
  }
  else
  {
    std::cout << "Unknown distortion model is provided." << std::endl;
    exit(-1);
  }
  // load look-up table
  for (size_t i = 0; i < sensor_size_.height * sensor_size_.width; i++)
  {
    precomputed_rectified_points_.col(i) = Eigen::Matrix<double, 2, 1>(
      RectCoordinates(i).x, RectCoordinates(i).y);
  }
  ROS_INFO("Undistorted-Rectified Look-Up Table has been computed.");
}

void TimeSurface::drawEvents(const EventArray::iterator& first, const EventArray::iterator& last,
      double& t0, double& t1, Eigen::Matrix4d& T_1_0, cv::Mat &out, cv::Mat &out_without)
{
  size_t n_events = 0;
  size_t n_events_without = 0;

  Eigen::Matrix<double, 2, Eigen::Dynamic> events;
  Eigen::Matrix<double, 2, Eigen::Dynamic> events_without;
  events.resize(2, last - first);
  events_without.resize(2, last - first);

  const int height = sensor_size_.height;
  const int width = sensor_size_.width;
  CHECK_EQ(out.rows, height);
  CHECK_EQ(out.cols, width);
  CHECK_EQ(out_without.rows, height);
  CHECK_EQ(out_without.cols, width);

  if(camera_matrix_.at<double>(cv::Point(0, 0)) < 1)
  {
    std::cout << "Camera intrinsic matrix is not initialized!!!" << std::endl;
    return;
  }
  Eigen::Matrix4d K;
  K << camera_matrix_.at<double>(cv::Point(0, 0)), 0., camera_matrix_.at<double>(0, 2), 0.,
       0., camera_matrix_.at<double>(cv::Point(1, 1)), camera_matrix_.at<double>(1, 2), 0.,
       0., 0., 1., 0.,
       0., 0., 0., 1.;

  Eigen::Matrix4d T = K * T_1_0.inverse() * K.inverse();
  // double depth = scene_depth_;

  bool do_motion_correction = true;

  double dt = 0;
  for(auto e = first; e != last; ++e)
  {
    if (n_events % 10 == 0)
    {
      dt = (t1 - e->ts.toSec()) / (t1 - t0);
      // std::cout <<std::setprecision(17)<< "dt === " << dt << " " << t1 << " " << t0 << " " << e->ts.toSec()<< std::endl;
    }

    Eigen::Vector4d f;
    Eigen::Vector4d f_without;
    f.head<2>() = dvs_keypoint_lut_.col(e->x + e->y * width);
    f[2] = 1.;
    f[3] = 1.;

    f_without.head<2>() = dvs_keypoint_lut_.col(e->x + e->y * width);
    f_without[2] = 1.;
    f_without[3] = 1.;

    // if(n_events % 50 == 0) 
    //   std::cout << "without " << std::endl << f << std::endl;

    if (do_motion_correction)
    {
      f = (1.f - dt) * f + dt * (T * f);
    }
    // if(n_events % 50 == 0) 
    //   std::cout << " with " <<std::endl<< f << std::endl;

    events.col(n_events++) = f.head<2>();
    events_without.col(n_events_without++) = f_without.head<2>();
  }
  // std::cout << "n_events = " << n_events << " " << n_events_without << std::endl;

  for (size_t i=0; i != n_events; ++i)
  {
    const Eigen::Vector2d& f = events.col(i);
    const Eigen::Vector2d& f_without = events_without.col(i);

    int x0 = std::floor(f[0]);
    int y0 = std::floor(f[1]);
    int x0_without = std::floor(f_without[0]);
    int y0_without = std::floor(f_without[1]);
    // std::cout << "f f_without = " << f[0] << " " << f_without[0] << " "<< f[1] << " " << f_without[1] << std::endl;
    // std::cout << "x0 y0 = " << x0 << " " << x0_without << " " << y0 << " " << y0_without << std::endl;

    if(x0 >= 0 && x0 < width-1 && y0 >= 0 && y0 < height-1)
    {
      // if(abs((f[0]-x0) - (float (f[0] - x0))) > 0.0001)
      // {
      //   std::cout << "!!!!=========" << (f[0] - x0) << "  " << float (f[0] - x0) << std::endl;
      // }
      const float fx = (float) (f[0] - x0);
      const float fy = (float) (f[1] - y0);
      Eigen::Vector4f w((1.f-fx)*(1.f-fy),
                        (fx)*(1.f-fy),
                        (1.f-fx)*(fy),
                        (fx)*(fy));

      out.at<float>(y0,   x0)   += w[0];
      out.at<float>(y0,   x0+1) += w[1];
      out.at<float>(y0+1, x0)   += w[2];
      out.at<float>(y0+1, x0+1) += w[3];
    }
    if(x0_without >= 0 && x0_without < width-1 && y0_without >= 0 && y0_without < height-1)
    {
      // std::cout << "=========" << (f[0] - x0) << "  " << (float) (f[0] - x0) << std::endl;
      const float fx = (float) (f_without[0] - x0_without);
      const float fy = (float) (f_without[1] - x0_without);
      Eigen::Vector4f w((1.f-fx)*(1.f-fy),
                        (fx)*(1.f-fy),
                        (1.f-fx)*(fy),
                        (fx)*(fy));

      out_without.at<float>(y0_without,   x0_without)   += w[0];
      out_without.at<float>(y0_without,   x0_without+1) += w[1];
      out_without.at<float>(y0_without+1, x0_without)   += w[2];
      out_without.at<float>(y0_without+1, x0_without+1) += w[3];
    }
  }

  // for (int y = 0; y < sensor_size_.height; y++)
  // {
  //   for (int x = 0; x < sensor_size_.width; x++)
  //   {
  // //     if(out.at<float>(x, y) != out_without.at<float>(x, y))
  // //     {
  // //       std::cout << "!!!!!!!!!!!!============" << std::endl;
  //       std::cout << "out xy  " << out.at<float>(x, y) << std::endl;
  // //     }
  //   }
  // }

}

void TimeSurface::mergeEvents(const EventArray::iterator& last, std::vector<int>& e_size, double* times_begin, double* times_end, Eigen::Matrix4d* T_delta, cv::Mat& out, cv::Mat& out_without)
{
  size_t n_events = 0;
  size_t n_events_without = 0;

  int total_events_size = 0;
  for(int i = 0; i < combine_frame_size_; i++)
  {
    total_events_size += e_size[i];
  }
  Eigen::Matrix<double, 2, Eigen::Dynamic> events;
  Eigen::Matrix<double, 2, Eigen::Dynamic> events_without;
  events.resize(2, total_events_size);
  events_without.resize(2, total_events_size);
  std::cout << "local time 1.4.1 " << boost::posix_time::microsec_clock::local_time() << std::endl;

  const int height = sensor_size_.height;
  const int width = sensor_size_.width;
  CHECK_EQ(out.rows, height);
  CHECK_EQ(out.cols, width);
  CHECK_EQ(out_without.rows, height);
  CHECK_EQ(out_without.cols, width);

  if(camera_matrix_.at<double>(cv::Point(0, 0)) < 1)
  {
    std::cout << "Camera intrinsic matrix is not initialized!!!" << std::endl;
    return;
  }
  Eigen::Matrix4d K;
  K << camera_matrix_.at<double>(cv::Point(0, 0)), 0., camera_matrix_.at<double>(0, 2), 0.,
       0., camera_matrix_.at<double>(cv::Point(1, 1)), camera_matrix_.at<double>(1, 2), 0.,
       0., 0., 1., 0.,
       0., 0., 0., 1.;

  bool do_motion_correction = true;

  // // e_size store accumulated events number. E.g. [0] stores event size in the end frame. [1] stores event size in the past two frames
  // for(int i = 0; i < combine_frame_size_; i++)
  // {
  //   const EventArray::iterator begin_idx = last - e_size[combine_frame_size_-i-1];
  //   const EventArray::iterator end_idx = last - e_size[combine_frame_size_-i-2];
  //   double t0 = times_begin[i];
  //   double t1 = times_end[i];
  //   int i_idx = i;
  //   Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
  //   while(i_idx < combine_frame_size_)
  //   {
  //     T = T * T_delta[i_idx];
  //     i_idx++;
  //   }
  //   std::cout << "T1 = " << std::endl << T << std::endl;
  //   T = K * T_delta[i].inverse() * K.inverse();
  //   std::cout << "T2 = " << std::endl << T << std::endl;

  //   double dt = 0;
  //   for(auto e = begin_idx; e != end_idx; ++e)
  //   {
  //     if (n_events % 10 == 0)
  //     {
  //       dt = (t1 - e->ts.toSec()) / (t1 - t0);
  //       // std::cout << "dt === " << dt << " " << t1 << " " << t0 << " " << e->ts.toSec()<< std::endl;
  //     }

  //     // double depth = scene_depth_;
  //     Eigen::Vector4d f;
  //     Eigen::Vector4d f_without;
  //     f.head<2>() = dvs_keypoint_lut_.col(e->x + e->y * width);
  //     f[2] = 1.;
  //     f[3] = 1.;

  //     f_without.head<2>() = dvs_keypoint_lut_.col(e->x + e->y * width);
  //     f_without[2] = 1.;
  //     f_without[3] = 1.;


  //     if (do_motion_correction)
  //     {
  //       f = (1.f - dt) * f + dt * (T * f);
  //     }

  //     events.col(n_events++) = f.head<2>();
  //     events_without.col(n_events_without++) = f_without.head<2>();
  //   }
  // }
          std::cout << "local time 1.4.2 " << boost::posix_time::microsec_clock::local_time() << std::endl;

  // e_size store accumulated events number. E.g. [0] stores event size in the end frame. [1] stores event size in the past two frames
  for(int i = 0; i < combine_frame_size_; i++)
  {
    const EventArray::iterator begin_idx = last - e_size[combine_frame_size_-i-1];
    const EventArray::iterator end_idx = last - e_size[combine_frame_size_-i-2];
    double t0 = times_begin[i];
    double t1 = times_end[i];
    

    Eigen::Matrix4d T = K * T_delta[i].inverse() * K.inverse();


    double dt = 0;
    int i_idx = i+1;
    Eigen::Matrix4d T_to_current = Eigen::Matrix4d::Identity();
    while(i_idx < combine_frame_size_)
    {
      T_to_current = T_to_current * T_delta[i_idx];
      i_idx++;
    }

    for(auto e = begin_idx; e != end_idx; ++e)
    {
      if (n_events % 10 == 0)
      {
        dt = (t1 - e->ts.toSec()) / (t1 - t0);
        // std::cout <<std::setprecision(17)<< "dt === " << dt << " " << t1 << " " << t0 << " " << e->ts.toSec()<< std::endl;
      }

      // double depth = scene_depth_;
      Eigen::Vector4d f;
      Eigen::Vector4d f_without;
      f.head<2>() = dvs_keypoint_lut_.col(e->x + e->y * width);
      f[2] = 1.;
      f[3] = 1.;

      f_without.head<2>() = dvs_keypoint_lut_.col(e->x + e->y * width);
      f_without[2] = 1.;
      f_without[3] = 1.;


      if (do_motion_correction)
      {
        f = T_to_current * ((1.f - dt) * f + dt * (T * f));
      }
      // if(n_events % 50 == 0) 
      //   std::cout << "i = " << i << std::endl << "without " << std::endl << ((1.f - dt) * f + dt * (T * f)) << std::endl << " with " <<std::endl<< f << std::endl;
      // if(abs(((1.f - dt) * f + dt * (T * f))[0]-f[0])>2) std::cout<< " = " << abs(((1.f - dt) * f + dt * (T * f))[0]-f[0]) << " !!!!Above 2 " << std::endl;


      events.col(n_events++) = f.head<2>();
      events_without.col(n_events_without++) = f_without.head<2>();
    }


  }
          std::cout << "local time 1.4.3 " << boost::posix_time::microsec_clock::local_time() << std::endl;

  for (size_t i=0; i != n_events; ++i)
  {
    const Eigen::Vector2d& f = events.col(i);
    const Eigen::Vector2d& f_without = events_without.col(i);

    int x0 = std::floor(f[0]);
    int y0 = std::floor(f[1]);
    int x0_without = std::floor(f_without[0]);
    int y0_without = std::floor(f_without[1]);
    // std::cout << "f f_without = " << f[0] << " " << f_without[0] << " "<< f[1] << " " << f_without[1] << std::endl;
    // std::cout << "x0 y0 = " << x0 << " " << x0_without << " " << y0 << " " << y0_without << std::endl;

    if(x0 >= 0 && x0 < width-1 && y0 >= 0 && y0 < height-1)
    {
      // if(abs((f[0]-x0) - (float (f[0] - x0))) > 0.0001)
      // {
      //   std::cout << "!!!!=========" << (f[0] - x0) << "  " << float (f[0] - x0) << std::endl;
      // }
      const float fx = (float) (f[0] - x0);
      const float fy = (float) (f[1] - y0);
      Eigen::Vector4f w((1.f-fx)*(1.f-fy),
                        (fx)*(1.f-fy),
                        (1.f-fx)*(fy),
                        (fx)*(fy));

      out.at<float>(y0,   x0)   += w[0];
      out.at<float>(y0,   x0+1) += w[1];
      out.at<float>(y0+1, x0)   += w[2];
      out.at<float>(y0+1, x0+1) += w[3];
    }
    if(x0_without >= 0 && x0_without < width-1 && y0_without >= 0 && y0_without < height-1)
    {
      // std::cout << "=========" << (f[0] - x0) << "  " << (float) (f[0] - x0) << std::endl;
      const float fx = (float) (f_without[0] - x0_without);
      const float fy = (float) (f_without[1] - x0_without);
      Eigen::Vector4f w((1.f-fx)*(1.f-fy),
                        (fx)*(1.f-fy),
                        (1.f-fx)*(fy),
                        (fx)*(fy));

      out_without.at<float>(y0_without,   x0_without)   += w[0];
      out_without.at<float>(y0_without,   x0_without+1) += w[1];
      out_without.at<float>(y0_without+1, x0_without)   += w[2];
      out_without.at<float>(y0_without+1, x0_without+1) += w[3];
    }
  }
          std::cout << "local time 1.4.4 " << boost::posix_time::microsec_clock::local_time() << std::endl;

  return;
}


void TimeSurface::eventsCallback(const dvs_msgs::EventArray::ConstPtr& msg)
{
std::cout << "local time 0.0 " << boost::posix_time::microsec_clock::local_time() << std::endl;
  // std::cout << "T_W_I_1 = " << std::endl << T_W_I_ << std::endl;
  std::lock_guard<std::mutex> lock(data_mutex_);
  if(msg->events.size()<1) return;

  // for(int i = 1; i < msg->events.size(); i++) {
  //   if(msg->events[i].ts.toSec() - msg->events[i-1].ts.toSec() > 0.001)
  //   std::cout <<"=============WARNING=================="<< msg->events[i].ts.toSec() << " " << msg->events[i-1].ts.toSec()<< std::endl;
  // }

  // std::cout << "bSensorInitialized_ = " << bSensorInitialized_ << " " << bCamInfoAvailable_ << std::endl;
  if(!bSensorInitialized_ || !bCamInfoAvailable_)
  {
    init(msg->width, msg->height);
    event_time_last_ = msg->events[0].ts.toSec();

    // const EventArrayPtr& events_ptr_last_ = std::make_shared<EventArray>();
    int event_size_last = msg->events.size();
    events_ptr_last_->resize(event_size_last);
    for(int i = 0; i < event_size_last; i++) {
      events_ptr_last_->at((uint32_t) i) = msg->events[i];
    }

    return;
  }
std::cout << "local time 0.1 " << boost::posix_time::microsec_clock::local_time() << std::endl;

  for(const dvs_msgs::Event& e : msg->events)
  {
    events_.push_back(e);
    int i = events_.size() - 2;
    // events_ size larger than 2 and events_'s timestamp is more recent than e's timestamp
    // Store all the events in a timesequence order
    // Most recent (newest) events are stored at the end of queue
    while(i >= 0 && events_[i].ts > e.ts)
    {
      events_[i+1] = events_[i];
      i--;
    }
    events_[i+1] = e;

    // Get most recent (newest) events and insert into eqMat_
    const dvs_msgs::Event& last_event = events_.back();
    pEventQueueMat_->insertEvent(last_event);
  }
std::cout << "local time 0.2 " << boost::posix_time::microsec_clock::local_time() << std::endl;

  // std::cout << "T_W_I_ 2= " << std::endl << T_W_I_ << std::endl;

  // if the size of event queue is larger than 5000000, then clear queue to fit 5000000
  // the number 5000000 is the number of totoal events at every pixel
  clearEventQueue();
std::cout << "local time 0.3 " << boost::posix_time::microsec_clock::local_time() << std::endl;

  // for(int i = 1; i < events_.size(); i++) {
  //   if(events_[i].ts.toSec() - events_[i-1].ts.toSec() > 0.001)
  //   std::cout << "********************WARNING****************"<< events_[i].ts.toSec() << " " << events_[i-1].ts.toSec()<< std::endl;
  // }

  past_ten_frames_events_size_.push_back(msg->events.size());
  while(past_ten_frames_events_size_.size()>10) 
  {
    past_ten_frames_events_size_.erase(past_ten_frames_events_size_.begin());
  }
  // std::cout << " past_ten_frames_events_size_ size + " << past_ten_frames_events_size_.size() << std::endl;

std::cout << "local time 0.4 " << boost::posix_time::microsec_clock::local_time() << std::endl;

  
  // std::cout <<"event_time_now  = " << event_time_now << std::endl;
  if(imus_.size()<2)
  {
    std::cout << "IMU vector's size is too small" << std::endl;
    return;
  }
  // std::cout << "T_W_I_3 = " << std::endl << T_W_I_ << std::endl;
  // std::cout << "event_time_last_ event_time_now = " << event_time_last_ << "  " << event_time_now  << " " <<  msg->events[0].ts<< std::endl;
  // Eigen::Matrix4d T_km1_k;
  // double event_time_now = msg->events[msg->events.size()-1].ts.toSec();
  // T_km1_k = integrateDeltaPose(event_time_last_, event_time_now);
  // T_km1_k(0,3) = 0; // Simply set translation vector to zero because the estimation of translation is not accurate
  // T_km1_k(1,3) = 0;
  // T_km1_k(2,3) = 0;
  // if(abs(euler_km1_k(0))<1 && abs(euler_km1_k(0))>0.5) return;
  // if(abs(euler_km1_k(1))<1 && abs(euler_km1_k(1))>0.5) return;
  // if(abs(euler_km1_k(2))<1 && abs(euler_km1_k(2))>0.5) return;
  // if(3.14-abs(euler_km1_k(0))<1 && 3.14-abs(euler_km1_k(0))>0.5) return;
  // if(3.14-abs(euler_km1_k(1))<1 && 3.14-abs(euler_km1_k(1))>0.5) return;
  // if(3.14-abs(euler_km1_k(2))<1 && 3.14-abs(euler_km1_k(2))>0.5) return;
  // double dt = event_time_now - event_time_last_;

  int event_size = events_.size();
  if(event_size < MAX_EVENT_QUEUE_LENGTH-10) return;
  const EventArrayPtr& events_ptr = std::make_shared<EventArray>();
  events_ptr->resize(event_size);
  for(int i = 0; i < event_size; i++) {
    events_ptr->at((uint32_t) i) = events_[i];
  }
std::cout << "local time 0.5 " << boost::posix_time::microsec_clock::local_time() << std::endl;

  // for(int i = 1; i < event_size; i++) {
  //   if(events_ptr->at(i).ts.toSec() - events_ptr->at(i-1).ts.toSec() > 0.001)
  //   std::cout << "!!!!!!!!!!!!!!!!!!!WARNING!!!!!!!!!!!!!!!!"<< events_ptr->at(i).ts.toSec() << " " << events_ptr->at(i-1).ts.toSec()<< std::endl;
  // }

  const EventArrayPtr& events_ptr_current = std::make_shared<EventArray>();
  int event_size_current = msg->events.size();
  events_ptr_current->resize(event_size_current);
  for(int i = 0; i < event_size_current; i++) {
    events_ptr_current->at((uint32_t) i) = msg->events[i];
  }

std::cout << "local time 0.6 " << boost::posix_time::microsec_clock::local_time() << std::endl;

  // std::vector<dvs_msgs::Event>::const_iterator first_event = msg->events.begin();
  // std::vector<dvs_msgs::Event>::const_iterator last_event = msg->events.end();

  int height = msg->height;
  int width = msg->width;

// if(!msg->empty())

  cv::Mat event_img = cv::Mat::zeros(height, width, CV_32F);
  cv::Mat event_img_without = cv::Mat::zeros(height, width, CV_32F);
  cv::Mat event_img01 = cv::Mat::zeros(height, width, CV_32F);
  cv::Mat event_img_without01 = cv::Mat::zeros(height, width, CV_32F);
  cv::Mat event_img12 = cv::Mat::zeros(height, width, CV_32F);
  cv::Mat event_img_without12 = cv::Mat::zeros(height, width, CV_32F);
  cv::Mat event_img23 = cv::Mat::zeros(height, width, CV_32F);
  cv::Mat event_img_without23 = cv::Mat::zeros(height, width, CV_32F);
  int noise_event_rate_ = 20000;
  int n_events_for_noise_detection = std::min(event_size, 2000);
  double event_rate = double (n_events_for_noise_detection) /
      (events_ptr->back().ts -
        events_ptr->at(event_size-n_events_for_noise_detection).ts).toSec();
std::cout << "local time 0.7 " << boost::posix_time::microsec_clock::local_time() << std::endl;

  // Only draw a new event image if the rate of events is sufficiently high
  // If not, then just use the previous drawn image in the backend.
  // This ensures that feature tracks are still tracked, and also induces
  // a subtile NoMotion prior to the backend: instead of directly using
  // NoMotion.
  if(event_rate > noise_event_rate_)
  {
    // Build event frame with fixed number of events
    const size_t winsize_events = 15000;
    int first_idx = std::max((int)event_size - (int) winsize_events, 0);

    if(event_size < winsize_events)
    {
      std::cout << "Requested frame size of length " << winsize_events
                    << " events, but I only have " << event_size
                    << " events in the last event array" << std::endl;
    }
    else
    {
      // visualizeEvents(events_ptr->begin(), events_ptr->end(), event_img); // + first_idx
      int event_size_last = events_ptr_last_->end()-events_ptr_last_->begin();
std::cout << "local time 0.8 " << boost::posix_time::microsec_clock::local_time() << std::endl;
      switch (projection_mode_)
      {
        case 0:
        {
          Eigen::Matrix4d T_km1_k;
          // double event_time_now = msg->events[msg->events.size()-1].ts.toSec();
          double e_time_begin = events_ptr->at(first_idx).ts.toSec();
          double e_time_end = (events_ptr->end()-1)->ts.toSec();   
          T_km1_k = integrateDeltaPose(e_time_begin, e_time_end);
          T_km1_k(0,3) = 0; // Simply set translation vector to zero because the estimation of translation is not accurate
          T_km1_k(1,3) = 0;
          T_km1_k(2,3) = 0;
          drawEvents(events_ptr->begin()+first_idx, events_ptr->end(), e_time_begin, e_time_end, T_km1_k, event_img, event_img_without);
        }
          break;
        case 1:
        {
          Eigen::Matrix4d T_km1_k;
          double event_time_now = msg->events[msg->events.size()-1].ts.toSec();
          T_km1_k = integrateDeltaPose(event_time_last_, event_time_now);
          T_km1_k(0,3) = 0; // Simply set translation vector to zero because the estimation of translation is not accurate
          T_km1_k(1,3) = 0;
          T_km1_k(2,3) = 0;
          const EventArrayPtr& events_ptr_combine = std::make_shared<EventArray>();
          // int event_size_last = events_ptr_last_->end()-events_ptr_last_->begin();
          int event_size_combine = event_size_last + events_ptr_current->end()-events_ptr_current->begin();
          // std::cout << "event_size_last  == " << event_size_last << std::endl;
          // std::cout << "event_size_current  == " << events_ptr_current->end()-events_ptr_current->begin() << std::endl;
          // std::cout << "event_size_combine  == " << event_size_combine << std::endl;
          events_ptr_combine->resize(event_size_combine);
          for(int i = 0; i < event_size_last; i++) 
          {
            events_ptr_combine->at((uint32_t) i) = events_ptr_last_->at((uint32_t) i);
          }
          for(int i = event_size_last; i < event_size_combine; i++)
          {
            events_ptr_combine->at((uint32_t) i) = events_ptr_current->at((uint32_t) (i-event_size_last));
          }
          drawEvents(events_ptr_combine->begin(), events_ptr_combine->end(), event_time_last_, event_time_now, T_km1_k, event_img, event_img_without);
        }
          break;
        case 2:
        {
          std::cout << "local time 1.0 " << boost::posix_time::microsec_clock::local_time() << std::endl;
          int frame_size = past_ten_frames_events_size_.size();
          int e_size_total = events_ptr->end()-events_ptr->begin();
          if(combine_frame_size_>9 || frame_size < 3) 
          {
            std::cout << "Combine_frame_size is too large, should be smaller than 9. Or event frame size is too small, should be larger than 2" << std::endl;
            return;
          }
          // if(e_size_total>e_size_last_three)
          // {
          //   std::cout << "Combine_frame_size is too large, should be smaller than 9" << std::endl;
          //   return;
          // }

          int combine_frame_size = combine_frame_size_;
          std::vector<int> e_size_accu_vec;
          int accumulated_events_size = 0;
          std::cout << "local time 1.1 " << boost::posix_time::microsec_clock::local_time() << std::endl;
          while(combine_frame_size>0)
          {
            frame_size--;
            accumulated_events_size += past_ten_frames_events_size_[frame_size];
            // store accumulated events number. E.g. [0] stores event size in the end frame. [1] stores event size in the past two frames
            e_size_accu_vec.push_back(accumulated_events_size);
            combine_frame_size--;
          }
// std::cout << "0" << std::endl;
          std::cout << "local time 1.2 " << boost::posix_time::microsec_clock::local_time() << std::endl;

          double* times_begin = new double[combine_frame_size_];
          double* times_end = new double[combine_frame_size_];
          Eigen::Matrix4d* T_delta = new Eigen::Matrix4d[combine_frame_size_];
          
          cv::Mat* event_img_vec = new cv::Mat[combine_frame_size_*2];

// std::cout << "1" << std::endl;
// std::cout << "combine_frame_size_"<<combine_frame_size_ << std::endl;
          std::cout << "local time 1.3 " << boost::posix_time::microsec_clock::local_time() << std::endl;
          for(int i = combine_frame_size_; i > 0;)
          {
// std::cout << "for begin" << std::endl;
// std::cout << "e_size_accu_vec size = " << e_size_accu_vec.size() << " " << i << std::endl;

            // std::cout << "i-2 === " << i-2 << " " << e_size_accu_vec[i-2] << std::endl;
            times_begin[combine_frame_size_-i] = events_ptr->at(e_size_total-e_size_accu_vec[i-1]).ts.toSec();
            if(i == 1)
            {
              times_end[combine_frame_size_-i] = events_ptr->at(e_size_total-e_size_accu_vec[i-2]-1).ts.toSec(); // Notes: i-2 can be -1, vector[-1] is zero
            }
            else
            {
              times_end[combine_frame_size_-i] = events_ptr->at(e_size_total-e_size_accu_vec[i-2]).ts.toSec(); // Notes: i-2 can be -1, vector[-1] is zero
            }
            // std::cout <<std::setprecision(17)<< "times_begin = " << times_begin[combine_frame_size_-i] << std::endl;
            // std::cout <<std::setprecision(17)<< "times_end = " << times_end[combine_frame_size_-i] << std::endl;
            // std::cout << "time = "<< std::setprecision(17) << (events_ptr->end()-e_size_accu_vec[i-1])->ts.toSec() << " " << (events_ptr->end()-e_size_accu_vec[i-2]-1)->ts.toSec()  << std::endl;
            T_delta[combine_frame_size_-i] = integrateDeltaPose(times_begin[combine_frame_size_-i], times_end[combine_frame_size_-i]);
            T_delta[combine_frame_size_-i](0,3) = 0; // Simply set translation vector to zero because the estimation of translation is not accurate
            T_delta[combine_frame_size_-i](1,3) = 0;
            T_delta[combine_frame_size_-i](2,3) = 0;
            // event_img_vec[(combine_frame_size_-i)*2] = cv::Mat::zeros(height, width, CV_32F);
            // event_img_vec[(combine_frame_size_-i)*2+1] = cv::Mat::zeros(height, width, CV_32F);
            // const EventArray::iterator begin_idx = events_ptr->end()- e_size_accu_vec[i-1];
            // const EventArray::iterator end_idx = events_ptr->end()-e_size_accu_vec[i-2];
            // // drawEvents(events_ptr->end()-e_size_accu_vec[i-1], events_ptr->end()-e_size_accu_vec[i-2], times_begin[combine_frame_size_-i], 
            // drawEvents(begin_idx, end_idx, times_begin[combine_frame_size_-i], times_end[combine_frame_size_-i], T_delta[combine_frame_size_-i], 
            //             event_img_vec[(combine_frame_size_-i)*2], event_img_vec[(combine_frame_size_-i)*2+1]);
            // cv::Mat event_image = cv::Mat::zeros(height, width, CV_32F);
            // event_image = event_img_vec[(combine_frame_size_-i)*2].clone();
            // cv::Mat event_image_without = cv::Mat::zeros(height, width, CV_32F);
            // event_image_without = event_img_vec[(combine_frame_size_-i)*2+1].clone();
            // cv::imshow(std::to_string(combine_frame_size_-i), event_image);
            // cv::imshow(std::to_string(combine_frame_size_-i)+"_without", event_image_without);
            i--;
// std::cout << "for end" << std::endl;
          }
          std::cout << "local time 1.4 " << boost::posix_time::microsec_clock::local_time() << std::endl;

          cv::Mat event_img_merge = cv::Mat::zeros(height, width, CV_32F);
          cv::Mat event_img_merge_without = cv::Mat::zeros(height, width, CV_32F);
          std::cout << "local time 1.4.0 " << boost::posix_time::microsec_clock::local_time() << std::endl;
          mergeEvents(events_ptr->end(), e_size_accu_vec, times_begin, times_end, T_delta, event_img_merge, event_img_merge_without);
          std::cout << "local time 1.4.9 " << boost::posix_time::microsec_clock::local_time() << std::endl;
          
          cv::Mat event_img_merge_clone = event_img_merge.clone();
          cv::Mat dilated, eroded_img;
          // cv::Mat event_img_merge_without_clone = event_img_merge_without.clone();
          cv::Mat win_size = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
          cv::dilate(event_img_merge_clone, dilated, win_size);
          cv::imshow("dilate", dilated);
          cv::erode(dilated, eroded_img, win_size);
          cv::imshow("erode", eroded_img);

          cv::imshow("merge event image", event_img_merge);
          cv::imshow("merge event image without", event_img_merge_without);
          cv::waitKey(1);
          std::cout << "local time 1.5 " << boost::posix_time::microsec_clock::local_time() << std::endl;

          // int ef_size = past_ten_frames_events_size_.size();
          // int e_size_last_one = past_ten_frames_events_size_[ef_size-1];
          // std::cout << "e_size_last_one " << e_size_last_one << " " << accumulated_events_size << std::endl;
          // int e_size_last_two = past_ten_frames_events_size_[ef_size-1]+past_ten_frames_events_size_[ef_size-2];
          // int e_size_last_three = past_ten_frames_events_size_[ef_size-1]+past_ten_frames_events_size_[ef_size-2]+past_ten_frames_events_size_[ef_size-3];
          // int event_size_total = events_ptr->end()-events_ptr->begin();
          // double e_time_begin_1 = events_ptr->at(event_size_total-e_size_last_three).ts.toSec();
          // double e_time_end_1 = events_ptr->at(event_size_total-e_size_last_two-1).ts.toSec();
          // double e_time_begin_2 = events_ptr->at(event_size_total-e_size_last_two).ts.toSec();
          // double e_time_end_2 = events_ptr->at(event_size_total-e_size_last_one-1).ts.toSec();
          // double e_time_begin_3 = events_ptr->at(event_size_total-e_size_last_one).ts.toSec();
          // double e_time_end_3 = (events_ptr->end()-1)->ts.toSec();
          // std::cout << "begin 1 = " << e_time_begin_3 << std::endl;
          // std::cout << "end 1 = " << e_time_end_3 << std::endl;
  //         double event_time_now = msg->events[msg->events.size()-1].ts.toSec();
  //         Eigen::Matrix4d T_delta_01 = integrateDeltaPose(e_time_begin_1, e_time_end_1);
  //         Eigen::Matrix4d T_delta_12 = integrateDeltaPose(e_time_begin_2, e_time_end_2);
  //         Eigen::Matrix4d T_delta_23 = integrateDeltaPose(e_time_begin_3, e_time_end_3);
  //         Eigen::Matrix4d T_delta = Eigen::Matrix4d::Identity();
  //         // T_delta = T_delta_01 * T_delta_12 * T_delta_23;
  //         // Eigen::Matrix4d T_delta = integrateDeltaPose(e_time_begin, e_time_end);
  //         // std::cout << "T_delta_01 = " << std::endl << T_delta_01 << std::endl;
  //         // std::cout << "T_delta_12 = " << std::endl << T_delta_12 << std::endl;
  //         // std::cout << "T_delta_23 = " << std::endl << T_delta_23 << std::endl;
  //         // std::cout << "T_delta_com = " << std::endl << T_delta_com << std::endl;
  //         // std::cout << "T_delta = " << std::endl << T_delta << std::endl;
  //         T_delta_01(0,3) = 0; // Simply set translation vector to zero because the estimation of translation is not accurate
  //         T_delta_01(1,3) = 0;
  //         T_delta_01(2,3) = 0;
  //         T_delta_12(0,3) = 0; // Simply set translation vector to zero because the estimation of translation is not accurate
  //         T_delta_12(1,3) = 0;
  //         T_delta_12(2,3) = 0;
  //         T_delta_23(0,3) = 0; // Simply set translation vector to zero because the estimation of translation is not accurate
  //         T_delta_23(1,3) = 0;
  //         T_delta_23(2,3) = 0;
  //         // T_delta(0,3) = 0; // Simply set translation vector to zero because the estimation of translation is not accurate
  //         // T_delta(1,3) = 0;
  //         // T_delta(2,3) = 0;

  //         // std::cout << "e_size_total  ============ " << e_size_total << "  " << e_size_3 << std::endl;
  //         // std::cout << "e_size_2 ============ " << past_ten_frames_events_size_[ef_size-1] << "  " << past_ten_frames_events_size_[ef_size-2] << std::endl;
  //         // std::cout << "event_size_last  == " << event_size_last << std::endl;
  //         // std::cout << "event_size_current  == " << events_ptr_current->end()-events_ptr_current->begin() << std::endl;
  //         if(ef_size>2 && e_size_total>e_size_last_three)
  //         {
  //           // std::cout << "should = " << std::setprecision(17)<< events_ptr->at(e_size_total-e_size_3-1).ts.toSec() << " " << (events_ptr->end()-e_size_3-1)->ts.toSec()<<std::endl;
  //           // std::cout << "should = " << std::setprecision(17)<< events_ptr->at(e_size_total-e_size_3).ts.toSec() <<" " << events_ptr_last_->begin()->ts.toSec()<< std::endl;
  //           // std::cout << "should = " << std::setprecision(17)<<e_time_end<<" "<< (events_ptr->end()-1)->ts.toSec() << std::endl;
  //           // std::cout << "should = " << std::setprecision(17)<< e_time_begin <<" " << (events_ptr->end()-e_size_3)->ts.toSec()<< std::endl;
  // //           std::cout << "T == " << std::endl << T_km1_k << std::endl << T_delta << std::endl;
  //           std::cout << "time1 = "<< std::setprecision(17) << (events_ptr->end()-e_size_last_three-1)->ts.toSec() << " " << (events_ptr->end()-e_size_last_two-1)->ts.toSec()  << std::endl;
  //           std::cout << "time2 = "<< std::setprecision(17) << e_time_begin_1 <<" " << e_time_end_1<< std::endl;
  //           drawEvents(events_ptr->end()-e_size_last_three, events_ptr->end()-e_size_last_two, e_time_begin_1, e_time_end_1, T_delta_01, event_img01, event_img_without01);
  //           drawEvents(events_ptr->end()-e_size_last_two, events_ptr->end()-e_size_last_one, e_time_begin_2, e_time_end_2, T_delta_12, event_img12, event_img_without12);
  //           drawEvents(events_ptr->end()-e_size_last_one, events_ptr->end(), e_time_begin_3, e_time_end_3, T_delta_23, event_img23, event_img_without23);
  //           // drawEvents(events_ptr->end()-e_size_3,
  //           //                   events_ptr->end(), e_time_begin, e_time_end, T_delta, event_img, event_img_without);
  //         }
  //         else
  //         {
  //           // drawEvents(events_ptr_combine->begin(), events_ptr_combine->end(), event_time_last_, event_time_now, T_km1_k, event_img, event_img_without);
  //           drawEvents(events_ptr_current->begin(), events_ptr_current->end(), event_time_last_, event_time_now, T_delta, event_img, event_img_without);
  //         }
        }
          break;
        default:
        {
          Eigen::Matrix4d T_km1_k;
          double event_time_now = msg->events[msg->events.size()-1].ts.toSec();
          T_km1_k = integrateDeltaPose(event_time_last_, event_time_now);
          T_km1_k(0,3) = 0; // Simply set translation vector to zero because the estimation of translation is not accurate
          T_km1_k(1,3) = 0;
          T_km1_k(2,3) = 0;
          drawEvents(events_ptr_current->begin(), events_ptr_current->end(), event_time_last_, event_time_now, T_km1_k, event_img, event_img_without);
        }
      }
      // std::cout << " =========" << event_img.at<float>(100, 100) << "  " << event_img_without.at<float>(100, 100) << std::endl;
      // for (int y = 0; y < sensor_size_.height; y++)
      // {
      //   for (int x = 0; x < sensor_size_.width; x++)
      //   {
      //     if(event_img.at<float>(x, y) != event_img_without.at<float>(x, y))
      //     {
      //       std::cout << "!!!!!!!!!!!!============" << std::endl;
      //       std::cout << event_img.at<float>(x, y) << "  " << event_img_without.at<float>(x, y) << std::endl;
      //     }
      //   }
      // }
      // cv::namedWindow("event_img", cv::WINDOW_AUTOSIZE);
      // cv::namedWindow("event_img_without", cv::WINDOW_AUTOSIZE);
      // cv::imshow("event_img", event_img);
      // cv::imshow("event_img_without", event_img_without);
      // cv::namedWindow("event_img01", cv::WINDOW_AUTOSIZE);
      // cv::namedWindow("event_img_without01", cv::WINDOW_AUTOSIZE);
      // cv::imshow("event_img01", event_img01);
      // cv::imshow("event_img_without01", event_img_without01);
      // cv::namedWindow("event_img12", cv::WINDOW_AUTOSIZE);
      // cv::namedWindow("event_img_without12", cv::WINDOW_AUTOSIZE);
      // cv::imshow("event_img12", event_img12);
      // cv::imshow("event_img_without12", event_img_without12);
      // cv::namedWindow("event_img23", cv::WINDOW_AUTOSIZE);
      // cv::namedWindow("event_img_without23", cv::WINDOW_AUTOSIZE);
      // cv::imshow("event_img23", event_img23);
      // cv::imshow("event_img_without23", event_img_without23);
      // cv::waitKey(1);
    }
  }
  // event_time_last_ = event_time_now;
  // event_time_last_ = events_ptr_last_->begin()->ts.toSec();
          std::cout << "local time 1.6 " << boost::posix_time::microsec_clock::local_time() << std::endl;
  events_ptr_last_->clear();
  event_size_current = events_ptr_current->end()-events_ptr_current->begin();
  events_ptr_last_->resize(event_size_current);
  for(int i = 0; i < event_size_current; i++) 
  {
    events_ptr_last_->at((uint32_t) i) = events_ptr_current->at((uint32_t) i);
  }
  event_time_last_ = events_ptr_last_->begin()->ts.toSec();
  events_ptr_current->clear();
          std::cout << "local time 1.7 " << boost::posix_time::microsec_clock::local_time() << std::endl;
}

void TimeSurface::clearEventQueue()
{
  // static constexpr size_t MAX_EVENT_QUEUE_LENGTH = 10000;
  // std::cout << "events size = " << events_.size() << std::endl;
  if (events_.size() > MAX_EVENT_QUEUE_LENGTH)
  {
    // std::cout << "events before size = " << events_.size() << std::endl;
    size_t remove_events = events_.size() - MAX_EVENT_QUEUE_LENGTH;
    events_.erase(events_.begin(), events_.begin() + remove_events);
    // std::cout << "events after size = " << events_.size() << std::endl;
  }
}

void TimeSurface::propagate(
      Eigen::Quaterniond& q,
      Eigen::Vector3d& p,
      Eigen::Vector3d& v,
      const Eigen::Vector3d& acc,
      const Eigen::Vector3d& gyr,
      const double dt)
{
  // Eigen::Quaterniond q_dt((gyr - gyr_bias_) * dt);
  // std::cout << "bias = " <<std::endl<< acc_bias_[0] << std::endl << gyr_bias_[0] << std::endl;
  Eigen::Vector3d ang_vel_unbias = (gyr - gyr_bias_) * dt;
  Eigen::Quaterniond q_delt = Eigen::AngleAxisd(ang_vel_unbias[2], Eigen::Vector3d::UnitZ()) * 
                  Eigen::AngleAxisd(ang_vel_unbias[1], Eigen::Vector3d::UnitY()) * 
                  Eigen::AngleAxisd(ang_vel_unbias[0], Eigen::Vector3d::UnitX());
  q = q * q_delt;
  v = v + (q.matrix()*(acc - acc_bias_)) * dt;
  p = p + v * dt;
  // v = v + (q.matrix()*(acc - acc_bias_) - g_) * dt;
  // std::cout << "q = " << std::endl << q.matrix() << std::endl; // Orientation is ok, but velocity and position is inaccurate.
  // std::cout << "v = " << std::endl << v << std::endl;
  // std::cout << "p = " << std::endl << p << std::endl;
}

Eigen::Matrix4d TimeSurface::integrateImu(Eigen::Matrix4d& T_Bkm1_W, Eigen::Vector3d& imu_linear_acc, Eigen::Vector3d& imu_angular_vel, 
                                          Eigen::Vector3d& tmp_V, const double& dt)
{
  Eigen::Matrix4d T_W_B = T_Bkm1_W.inverse();
  Eigen::Matrix3d R_W_B = T_W_B.block(0, 0, 3, 3);
  Eigen::Quaterniond q(R_W_B);
  Eigen::Vector3d t = T_W_B.block(0,3,3,1);
  propagate(q, t, tmp_V, imu_linear_acc, imu_angular_vel, dt);
  Eigen::Matrix4d T_W_B_new = Eigen::Matrix4d::Identity();
  T_W_B_new.block(0,0,3,3) = q.matrix();
  T_W_B_new.block(0,3,3,1) = t;
  return T_Bkm1_W * T_W_B_new;
}

Eigen::Matrix4d TimeSurface::integrateDeltaPose(double& t1, double& t2)
{
  // std::cout <<"begin =============== begin" << std::endl;
  int imu_size = imus_.size();
// std::cout <<"imu_size" << imu_size << std::endl;
  // std::cout<< std::setprecision(17) << t2 <<" " << t1 <<std::endl;
  // CHECK_EQ(t1, t2); TODO check less
  Eigen::Vector3d imu_linear_acc_t2, imu_angular_vel_t2, imu_linear_acc_t1, imu_angular_vel_t1, imu_linear_acc, imu_angular_vel;
  for(int i = imu_size; i > 0; i--)
  {
          // std::cout << t2 << std::endl;
      // std::cout <<"iiiiiiiiiiiiii" << std::setprecision(17)<< i << " " <<imus_[i-1].header.stamp.toSec()<<std::endl;
    if(imus_[i-1].header.stamp.toSec()<t2)
    {
      imu_linear_acc_t2 << imus_[i-1].linear_acceleration.x, imus_[i-1].linear_acceleration.y, imus_[i-1].linear_acceleration.z;
      imu_angular_vel_t2 << imus_[i-1].angular_velocity.x, imus_[i-1].angular_velocity.y, imus_[i-1].angular_velocity.z;
      // std::cout << "iii = " << i << std::endl;
      break;
    }
  }

  for(int i = imu_size; i > 0; i--)
  {
      // std::cout << t1 << std::endl;
      // std::cout <<"jjjjjjjjjjj" << std::setprecision(17)<< i  << " " <<imus_[i-1].header.stamp.toSec()<< std::endl;
    if(imus_[i-1].header.stamp.toSec()<t1)
    {
      imu_linear_acc_t1 << imus_[i-1].linear_acceleration.x, imus_[i-1].linear_acceleration.y, imus_[i-1].linear_acceleration.z;
      imu_angular_vel_t1 << imus_[i-1].angular_velocity.x, imus_[i-1].angular_velocity.y, imus_[i-1].angular_velocity.z;
      // std::cout << "jjj = " << i << std::endl;
      break;
    }
  }
  imu_linear_acc = (imu_linear_acc_t2+imu_linear_acc_t1)/2.0;
  imu_angular_vel = (imu_angular_vel_t2+imu_angular_vel_t1)/2.0;
  // std::cout << "imu linear angular = " << std::endl << imu_linear_acc << std::endl << imu_angular_vel << std::endl;

  // std::cout << "bias = " <<std::endl<< acc_bias_ << std::endl << gyr_bias_ << std::endl;
  // std::cout<<imu_angular_vel(2)<<std::endl;
  const double dt =(double) (t2 - t1);
  // std::cout << "dt = " << dt << std::endl;

  // Eigen::Matrix3d R_Bkm1_Bk_ = T_Bkm1_Bk_.block(0, 0, 3, 3);
  // Eigen::Quaterniond Q_Bkm1_Bk_(R_Bkm1_Bk_);
  // tmp_Q = tmp_Q * Q_Bkm1_Bk_;

  // Eigen::Vector3d acc= tmp_Q*(imu_linear_acc-acc_bias_);
  // tmp_V += (tmp_Q*(imu_linear_acc-acc_bias_))*dt;
  // tmp_P = tmp_P + tmp_Q* tmp_V*dt+0.5*dt*dt*acc;

  // Eigen::Vector3d euler_init(-3.1415926/2, 0, 0);

  // Eigen::Matrix3d R_init;
  // R_init = Eigen::AngleAxisd(euler_init[0], Eigen::Vector3d::UnitZ()) * 
  //                    Eigen::AngleAxisd(euler_init[1], Eigen::Vector3d::UnitY()) * 
  //                    Eigen::AngleAxisd(euler_init[2], Eigen::Vector3d::UnitX());
  Eigen::Matrix4d T_B_W = T_W_I_.inverse(); // Transformation matrix from world to body, TODO replace with real matrix
  int pose_size = T_W_I_vec_.size();
  // std::cout << "pose_size = " << pose_size << std::endl;
  if(pose_size>2)
  {
    vel_W = (T_W_I_vec_[pose_size-1].block(0,3,3,1) - T_W_I_vec_[pose_size-2].block(0,3,3,1))/dt;
  }
  else
  {
    vel_W << 0.0001,0.0001,0.0001;
  }
  // std::cout << "vel = " << vel_W << std::endl;
  // vel_W += (imu_linear_acc-acc_bias_)*dt;
  // std::cout << "T_B_W 1 = " << std::endl << T_B_W << std::endl;
  T_Bkm1_Bk_ = integrateImu(T_B_W, imu_linear_acc, imu_angular_vel, vel_W, dt);
  // std::cout << "T_Bkm1_Bk_() = "<< std::endl << T_Bkm1_Bk_ << std::endl;
  // std::cout << "T_Bkm1_Bk_.inverse() = "<< std::endl << T_Bkm1_Bk_.inverse() << std::endl;
  // std::cout << "T_B_W 2 = " << std::endl << T_B_W << std::endl;
  Eigen::Matrix4d T_I_W_ = T_Bkm1_Bk_.inverse() * T_B_W;
  // std::cout << "T_W_I_ 3.3= " << std::endl << T_W_I_ << std::endl;
  // std::cout << "T_I_W_= " << std::endl << T_I_W_ << std::endl;
  T_W_I_ = T_I_W_.inverse();
  // std::cout << "T_W_I_ 3.4= " << std::endl << T_W_I_ << std::endl;
  T_W_I_vec_.push_back(T_W_I_);

  return T_Bkm1_Bk_;
}

void TimeSurface::imuCallback(const sensor_msgs::Imu::ConstPtr &msg)
{
  sensor_msgs::Imu imu;
  imu.header = msg->header;
  imu.orientation = msg->orientation;
  imu.angular_velocity = msg->angular_velocity;
  imu.linear_acceleration = msg->linear_acceleration;
  imus_.push_back(imu);

  clearImuVector();


  if(!imu_inited_)
  {
    imu_cnt_++;
    if(imu_cnt_ < 10)
    {
      return;
    }
    else if(imu_cnt_ >= 10 && imu_cnt_ < 50)
    {
      Eigen::Vector3d imu_linear_acc(msg->linear_acceleration.x, msg->linear_acceleration.y, msg->linear_acceleration.z);
      Eigen::Vector3d imu_angular_vel(msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z);
      acc_bias_ += imu_linear_acc;
      gyr_bias_ += imu_angular_vel;

      return;
    }
    acc_bias_ /= (imu_cnt_-10);
    gyr_bias_ /= (imu_cnt_-10);

    imu_inited_ = true;
    imu_time_last_ = msg->header.stamp.toSec();
    return;
  }

  Eigen::Matrix4d T_km1_k;
  double imu_time_now = msg->header.stamp.toSec();
  // T_km1_k =  integrateDeltaPose(imu_time_last_, imu_time_now); // DON'T use this in motion correction mode
  // std::cout << "T_km1_k = " <<std::endl<< T_km1_k << std::endl;
  imu_time_last_ = imu_time_now;

  // std::cout << "T_W_I_ = " << T_W_I_ << std::endl;

  Eigen::Matrix3d R_imu_ = T_W_I_.block(0,0,3,3);
  Eigen::Quaterniond quaternion_imu(R_imu_);
  //pub path
  geometry_msgs::PoseStamped this_pose_stamped;

  this_pose_stamped.pose.position.x = T_W_I_(0,3);
  this_pose_stamped.pose.position.y = T_W_I_(1,3);
  this_pose_stamped.pose.position.z = T_W_I_(2,3);
  this_pose_stamped.pose.orientation.x = quaternion_imu.x();
  this_pose_stamped.pose.orientation.y = quaternion_imu.y();
  this_pose_stamped.pose.orientation.z = quaternion_imu.z();
  this_pose_stamped.pose.orientation.w = quaternion_imu.w();

  // this_pose_stamped.pose.position.x = tmp_P(0);
  // this_pose_stamped.pose.position.y = tmp_P(1);
  // this_pose_stamped.pose.position.z = tmp_P(2);
  // this_pose_stamped.pose.orientation.x = tmp_Q.x();
  // this_pose_stamped.pose.orientation.y = tmp_Q.y();
  // this_pose_stamped.pose.orientation.z = tmp_Q.z();
  // this_pose_stamped.pose.orientation.w = tmp_Q.w();

  this_pose_stamped.header.stamp= ros::Time::now();
  this_pose_stamped.header.frame_id="map";



  // std::cout<<" dt: " <<  dt << std::cout<<"x="<<tmp_P(0)<<" y="<<tmp_P(1)<<" z="<<tmp_P(2)<<std::endl;

  g_imu_path.poses.push_back(this_pose_stamped);
  g_imu_path_pub.publish(g_imu_path);

  localizationPosePub_.publish(this_pose_stamped);

}

void TimeSurface::clearImuVector()
{
  static constexpr size_t MAX_IMU_VECTOR_LENGTH = 1000;
  // std::cout << "imu size before + " << imus_.size() << std::endl;
  if (imus_.size() > MAX_IMU_VECTOR_LENGTH)
  {
    size_t remove_imus = imus_.size() - MAX_IMU_VECTOR_LENGTH;
    imus_.erase(imus_.begin(), imus_.begin() + remove_imus);
  }
  // std::cout << "imu size after + " << imus_.size() << std::endl;
  // std::cout << imus_.end()->angular_velocity.x << std::endl;
}

} // namespace esvo_time_surface
