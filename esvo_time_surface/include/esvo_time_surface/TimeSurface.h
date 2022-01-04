#ifndef esvo_time_surface_H_
#define esvo_time_surface_H_

#include <ros/ros.h>
#include <std_msgs/Time.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/image_encodings.h>
#include "sensor_msgs/Imu.h"
#include <nav_msgs/Path.h>
#include <std_msgs/String.h>
#include <geometry_msgs/Quaternion.h>
#include <geometry_msgs/PoseStamped.h>
#include <dynamic_reconfigure/server.h>
#include <image_transport/image_transport.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <dvs_msgs/Event.h>
#include <dvs_msgs/EventArray.h>

#include <deque>
#include <mutex>
#include <Eigen/Eigen>

namespace esvo_time_surface
{
#define NUM_THREAD_TS 1
using EventQueue = std::deque<dvs_msgs::Event>;
using real_t = double;

class EventQueueMat 
{
public:
  EventQueueMat(int width, int height, int queueLen)
  {
    width_ = width;
    height_ = height;
    queueLen_ = queueLen;
    eqMat_ = std::vector<EventQueue>(width_ * height_, EventQueue());
  }

  void insertEvent(const dvs_msgs::Event& e)
  {
    if(!insideImage(e.x, e.y))
      return;
    else
    {
      EventQueue& eq = getEventQueue(e.x, e.y);
      eq.push_back(e); // Most recent event is at back
      while(eq.size() > queueLen_)
        eq.pop_front(); // remove oldest events to fit queue length
    }
  }

  bool getMostRecentEventBeforeT(
    const size_t x,
    const size_t y,
    const ros::Time& t,
    dvs_msgs::Event* ev)
  {
    // Outside of image: false
    if(!insideImage(x, y))
      return false;

    // No event at xy: false
    EventQueue& eq = getEventQueue(x, y);
    if(eq.empty())
      return false;

    // Loop through all events to find most recent event
    // Assume events are ordered from latest to oldest
    for(auto it = eq.rbegin(); it != eq.rend(); ++it)
    {
      const dvs_msgs::Event& e = *it;
      if(e.ts < t)
      {
        *ev = *it;
        return true;
      }
    }
    return false;
  }

  void clear()
  {
    eqMat_.clear();
  }

  bool insideImage(const size_t x, const size_t y)
  {
    return !(x < 0 || x >= width_ || y < 0 || y >= height_);
  }

  inline EventQueue& getEventQueue(const size_t x, const size_t y)
  {
    return eqMat_[x + width_ * y];
  }

  size_t width_;
  size_t height_;
  size_t queueLen_;
  std::vector<EventQueue> eqMat_;
};

class TimeSurface
{
  struct Job
  {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    EventQueueMat* pEventQueueMat_;
    cv::Mat* pTimeSurface_;
    size_t start_col_, end_col_;
    size_t start_row_, end_row_;
    size_t i_thread_;
    ros::Time external_sync_time_;
    double decay_sec_;
  };

public:
  TimeSurface(ros::NodeHandle & nh, ros::NodeHandle nh_private);
  virtual ~TimeSurface();

private:
  ros::NodeHandle nh_;
  // core
  void init(int width, int height);
  void createTimeSurfaceAtTime(const ros::Time& external_sync_time);// single thread version (This is enough for DAVIS240C and DAVIS346)
  void createTimeSurfaceAtMostRecentEvent();// single thread version (This is enough for DAVIS240C and DAVIS346)
  void createTimeSurfaceAtTime_hyperthread(const ros::Time& external_sync_time); // hyper thread version (This is for higher resolution)
  void thread(Job& job);

  // callbacks
  void syncCallback(const std_msgs::TimeConstPtr& msg);
  void eventsCallback(const dvs_msgs::EventArray::ConstPtr& msg);
  void cameraInfoCallback(const sensor_msgs::CameraInfo::ConstPtr& msg);
  void imuCallback(const sensor_msgs::Imu::ConstPtr &msg);

  // utils
  void clearEventQueue();
  void clearImuVector();
  Eigen::Matrix4d integrateImu(Eigen::Matrix4d& T_B_W, Eigen::Vector3d& imu_linear_acc, Eigen::Vector3d& imu_angular_vel, 
                              Eigen::Vector3d& tmp_V, const double& dt);
  void propagate(Eigen::Quaterniond& q, Eigen::Vector3d& p, Eigen::Vector3d& v, const Eigen::Vector3d& acc, 
                  const Eigen::Vector3d& gyr, const double dt);

  // calibration parameters
  cv::Mat camera_matrix_, dist_coeffs_;
  cv::Mat rectification_matrix_, projection_matrix_;
  std::string distortion_model_;
  cv::Mat undistort_map1_, undistort_map2_;
  Eigen::Matrix2Xd precomputed_rectified_points_;

  // sub & pub
  ros::Subscriber event_sub_;
  ros::Subscriber camera_info_sub_;
  ros::Subscriber sync_topic_;
  ros::Subscriber imu_sub_;
  image_transport::Publisher time_surface_pub_;
	ros::Publisher localizationPosePub_;

  // online parameters
  bool bCamInfoAvailable_;
  bool bUse_Sim_Time_;  
  cv::Size sensor_size_;
  ros::Time sync_time_;
  bool bSensorInitialized_;

  // offline parameters
  double decay_ms_;
  bool time_surface_at_most_recent_event_;
  bool ignore_polarity_;
  int median_blur_kernel_size_;
  int max_event_queue_length_;
  int events_maintained_size_;

  // containers
  EventQueue events_;
  std::shared_ptr<EventQueueMat> pEventQueueMat_;

  std::vector<sensor_msgs::Imu> imus_;

  // thread mutex
  std::mutex data_mutex_;

  clock_t current_time,init_time;
  double g_last_imu_time;
  bool imu_inited_;
  double imu_cnt_;
  Eigen::Vector3d acc_bias_;
  Eigen::Vector3d gyr_bias_;
  Eigen::Vector3d g_;
  ros::Time time_last_;
  Eigen::Vector3d tmp_P; //t
  Eigen::Quaterniond tmp_Q;//R
  Eigen::Vector3d tmp_V;
  ros::Publisher g_imu_path_pub;
  nav_msgs::Path g_imu_path;
  Eigen::Matrix4d T_imu_ = Eigen::Matrix4d::Identity();
  Eigen::Matrix4d T_Bkm1_Bk_ = Eigen::Matrix4d::Identity();


  // Time Surface Mode
  // Backward: First Apply exp decay on the raw image plane, then get the value
  //           at each pixel in the rectified image plane by looking up the
  //           corresponding one (float coordinates) with bi-linear interpolation.
  // Forward: First warp the raw events to the rectified image plane, then
  //          apply the exp decay on the four neighbouring (involved) pixel coordinate.
  enum TimeSurfaceMode
  {
    BACKWARD,// used in the T-RO20 submission
    FORWARD
  } time_surface_mode_;
};
} // namespace esvo_time_surface
#endif // esvo_time_surface_H_