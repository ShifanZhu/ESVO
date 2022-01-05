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
  // setup subscribers and publishers
  event_sub_ = nh_.subscribe("events", 0, &TimeSurface::eventsCallback, this);
  camera_info_sub_ = nh_.subscribe("camera_info", 1, &TimeSurface::cameraInfoCallback, this);
  sync_topic_ = nh_.subscribe("sync", 1, &TimeSurface::syncCallback, this);
  imu_sub_ = nh_.subscribe("/imu/data", 1, &TimeSurface::imuCallback, this);
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
                  std::cout << "reach one vu" << std::endl;
                  time_surface_map.at<double>(v_i, u_i) = 1;
                }  
                if(time_surface_map.at<double>(v_i, u_i + 1) > 1)
                {

                  std::cout << "reach one vu+1" << std::endl;
                  time_surface_map.at<double>(v_i, u_i + 1) = 1;
                }
                if(time_surface_map.at<double>(v_i + 1, u_i) > 1)
                {
                  std::cout << "reach one v+1u" << std::endl;
                  time_surface_map.at<double>(v_i + 1, u_i) = 1;
                }
                if(time_surface_map.at<double>(v_i + 1, u_i + 1) > 1)
                {
                  std::cout << "reach one v+1u+1" << std::endl;
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

  cv::Size sensor_size(msg->width, msg->height);
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
    //                                      sensor_size, CV_32FC1, undistort_map1_, undistort_map2_);
    bCamInfoAvailable_ = true;
    ROS_INFO("Camera information is loaded (Distortion model %s).", distortion_model_.c_str());
  }
  else if(distortion_model_ == "plumb_bob")
  {
    // cv::initUndistortRectifyMap(camera_matrix_, dist_coeffs_,
    //                             rectification_matrix_, projection_matrix_,
    //                             sensor_size, CV_32FC1, undistort_map1_, undistort_map2_);
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
  precomputed_rectified_points_ = Eigen::Matrix2Xd(2, sensor_size.height * sensor_size.width);
  // raw coordinates
  cv::Mat_<cv::Point2f> RawCoordinates(1, sensor_size.height * sensor_size.width);
  for (int y = 0; y < sensor_size.height; y++)
  {
    for (int x = 0; x < sensor_size.width; x++)
    {
      int index = y * sensor_size.width + x;
      RawCoordinates(index) = cv::Point2f((float) x, (float) y);
    }
  }
  // undistorted-rectified coordinates
  cv::Mat_<cv::Point2f> RectCoordinates(1, sensor_size.height * sensor_size.width);
  for(int i = 0; i < sensor_size.height * sensor_size.width; ++i)
  {
    RectCoordinates(i) = RawCoordinates(i);
  }
  std::cout << "direct apply points" << std::endl;
  if (distortion_model_ == "plumb_bob")
  {
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
  for (size_t i = 0; i < sensor_size.height * sensor_size.width; i++)
  {
    precomputed_rectified_points_.col(i) = Eigen::Matrix<double, 2, 1>(
      RectCoordinates(i).x, RectCoordinates(i).y);
  }
  ROS_INFO("Undistorted-Rectified Look-Up Table has been computed.");
}

void TimeSurface::eventsCallback(const dvs_msgs::EventArray::ConstPtr& msg)
{
  std::lock_guard<std::mutex> lock(data_mutex_);

  if(!bSensorInitialized_)
    init(msg->width, msg->height);

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

  // if the size of event queue is larger than 5000000, then clear queue to fit 5000000
  // the number 5000000 is the number of totoal events at every pixel
  clearEventQueue();
}

void TimeSurface::clearEventQueue()
{
  static constexpr size_t MAX_EVENT_QUEUE_LENGTH = 500000;
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
  std::cout << "acc - acc_bias_ = " << std::endl << acc - acc_bias_ << std::endl;
  std::cout << "(q.matrix()*(acc - acc_bias_)) = " << std::endl << (q.matrix()*(acc - acc_bias_)) << std::endl;
  std::cout << "(q.matrix()*(acc - acc_bias_))  * dt= " << std::endl << (q.matrix()*(acc - acc_bias_)) * dt << std::endl;
  std::cout << "v = " << std::endl << v << std::endl;
  std::cout << "p = " << std::endl << p << std::endl;
}

Eigen::Matrix4d TimeSurface::integrateImu(Eigen::Matrix4d& T_Bkm1_W, Eigen::Vector3d& imu_linear_acc, Eigen::Vector3d& imu_angular_vel, 
                                          Eigen::Vector3d& tmp_V, const double& dt)
{
  Eigen::Matrix4d T_W_B = T_Bkm1_W.inverse();
  Eigen::Matrix3d R_W_B = T_W_B.block(0, 0, 3, 3);
  Eigen::Quaterniond q(R_W_B);
  Eigen::Vector3d t = T_W_B.block(0,3,3,1);
  propagate(q, t, tmp_V, imu_linear_acc, imu_angular_vel, dt);
  Eigen::Matrix4d T_W_B_new;
  T_W_B_new.block(0,0,3,3) = q.matrix();
  T_W_B_new.block(0,3,3,1) = t;
  return T_Bkm1_W * T_W_B_new;
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
  int imu_size = imus_.size();
  Eigen::Vector3d imu_linear_acc(imus_[imu_size-1].linear_acceleration.x, imus_[imu_size-1].linear_acceleration.y, imus_[imu_size-1].linear_acceleration.z);
  Eigen::Vector3d imu_angular_vel(imus_[imu_size-1].angular_velocity.x, imus_[imu_size-1].angular_velocity.y, imus_[imu_size-1].angular_velocity.z);

  if(!imu_inited_)
  {
    imu_cnt_++;
    if(imu_cnt_ < 10)
    {
      return;
    }
    else if(imu_cnt_ >= 10 && imu_cnt_ < 50)
    {
      acc_bias_ += imu_linear_acc;
      gyr_bias_ += imu_angular_vel;

      return;
    }
    acc_bias_ /= (imu_cnt_-10);
    gyr_bias_ /= (imu_cnt_-10);

    imu_inited_ = true;
    time_last_ = msg->header.stamp;
    std::cout << "!!!not inited" << std::endl;
    return;
  }
  // std::cout << "bias = " <<std::endl<< acc_bias_ << std::endl << gyr_bias_ << std::endl;
  // std::cout<<imu_angular_vel(2)<<std::endl;
  const double dt =(double) (msg->header.stamp.toSec() - time_last_.toSec());
  // std::cout << "dt = " << dt << std::endl;
  time_last_ = msg->header.stamp;

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
  if(pose_size>2)
  {
    vel_W = (T_W_I_vec_[pose_size-1].block(0,3,3,1) - T_W_I_vec_[pose_size-2].block(0,3,3,1))/dt;
  }
  else
  {
    vel_W << 0.0001,0.0001,0.0001;
  }
  // vel_W += (imu_linear_acc-acc_bias_)*dt;
  T_Bkm1_Bk_ = integrateImu(T_B_W, imu_linear_acc, imu_angular_vel, vel_W, dt);
  // std::cout << "T_Bkm1_Bk_ = " << T_Bkm1_Bk_.inverse() << std::endl;
  Eigen::Matrix4d T_I_W_ = T_Bkm1_Bk_.inverse() * T_B_W;
  T_W_I_ = T_I_W_.inverse();
  T_W_I_vec_.push_back(T_W_I_);

  drawEvents(
            events_ptr->begin()+first_idx,
            events_ptr->end(),
            t0, t1,
            T_1_0,
            event_img);
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
  static constexpr size_t MAX_IMU_VECTOR_LENGTH = 100;
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
