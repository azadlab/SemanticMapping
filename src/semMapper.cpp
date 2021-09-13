/*

Author Name: J. Rafid S.
Author URI: www.azaditech.com
Relevant Publication: Siddiqui, J. R. , Andreasson, H. , Driankov, D. & Lilienthal, A. J. (2016). Towards visual mapping in industrial environments: a heterogeneous task-specific and saliency driven approach. IEEE International Conference on Robotics and Automation (ICRA). , Stockholm, Sweden, May, 2016.

*/

#include <semantic_mapping/semviz.h>
#include <image_transport/image_transport.h>
#include <image_transport/subscriber_filter.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <image_geometry/pinhole_camera_model.h>
#include <pcl/conversions.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/PointCloud2.h>
#include <nav_msgs/Odometry.h>
#include <iostream>
#include <semantic_mapping/semanticextractor.h>
#include <cv_bridge/cv_bridge.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <eigen_conversions/eigen_msg.h>
#include <sys/resource.h>
#include <std_srvs/Empty.h>

#define WRITE_RUNTIME

using namespace message_filters::sync_policies;
namespace enc = sensor_msgs::image_encodings;
typedef ApproximateTime<sensor_msgs::PointCloud2,nav_msgs::Odometry> VeloOdom;
typedef ApproximateTime<sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::CameraInfo,sensor_msgs::CameraInfo,nav_msgs::Odometry> KinectOdom;
typedef ApproximateTime<sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::CameraInfo,sensor_msgs::CameraInfo,sensor_msgs::PointCloud2,nav_msgs::Odometry> KinectVeloOdom;


typedef union
{
  struct
  {
    unsigned char Blue;
    unsigned char Green;
    unsigned char Red;
    unsigned char Alpha;
  };
  float float_value;
  long long_value;
} RGBValue;



class SemMapper {

    semanticExtractor *semEx;
    SemMapPtr semMap;

    boost::mutex m;
    Eigen::Affine3d pose;

    Eigen::Matrix4d Tcam_to_sensor;
    Eigen::Matrix4d Tsensor_to_rob;
    Eigen::Matrix4d Trob_to_world;
    Eigen::Matrix4d Tcam_to_world;
    ros::NodeHandle nh_;
    boost::shared_ptr<image_transport::ImageTransport> rgb_node_ptr, depth_node_ptr;
    //ros::Subscriber rob_state_sub;
    //boost::shared_ptr<KinectSync> sync_;
    //image_transport::SubscriberFilter sub_depth_, sub_rgb_;
    message_filters::Synchronizer<KinectVeloOdom> *kvsync;
    message_filters::Synchronizer<KinectOdom> *ksync;
    message_filters::Synchronizer<VeloOdom> *vsync;
    message_filters::Subscriber<sensor_msgs::Image> *sub_depth,*sub_rgb;
    message_filters::Subscriber<sensor_msgs::CameraInfo> *depth_info,*rgb_info;
    message_filters::Subscriber<sensor_msgs::PointCloud2> *sub_velo_points;
    message_filters::Subscriber<nav_msgs::Odometry> *sub_rob_state;
    image_geometry::PinholeCameraModel model_;
    ros::ServiceServer sem_saver;

    string mapPath;
    std::vector<string> models;
    std::vector<ObjectDetector> detectors;

    SemViz *viewer;
    double sensor_pose_x,sensor_pose_y,sensor_pose_z,sensor_pose_t;
    bool useKinect,visualize,isPoseInit;
    int nframes=0;
    int mapRegions=0;

    ofstream ofs;

public:

    SemMapper(ros::NodeHandle nh)
    {
        semMap = boost::make_shared<SemMap>(0.2,100);
        semEx = new semanticExtractor(semMap);
        nh.param("useKinect",useKinect,true);
        nh.param("visualize",visualize,false);
        nh.param("sensor_pose_x",sensor_pose_x,0.);
        nh.param("sensor_pose_y",sensor_pose_y,0.);
        nh.param("sensor_pose_z",sensor_pose_z,0.);
        nh.param("sensor_pose_t",sensor_pose_t,0.);
        nh.param("save_path",mapPath,string(""));

        sub_rob_state = new message_filters::Subscriber<nav_msgs::Odometry>(nh_,"/vmc_navserver/state",10);
        sub_velo_points = new message_filters::Subscriber<sensor_msgs::PointCloud2>(nh_,"/velodyne_points",1);
        if(useKinect)
        {
            sub_depth = new message_filters::Subscriber<sensor_msgs::Image>(nh_,"/camera/depth/image_raw",1);
            sub_rgb = new message_filters::Subscriber<sensor_msgs::Image>(nh_,"/camera/rgb/image_color",1);
            depth_info = new message_filters::Subscriber<sensor_msgs::CameraInfo>(nh_,"/camera/depth/camera_info",1);
            rgb_info = new message_filters::Subscriber<sensor_msgs::CameraInfo>(nh_,"/camera/rgb/camera_info",1);
            ksync =  new message_filters::Synchronizer<KinectOdom>(KinectOdom(2), *sub_depth, *sub_rgb, *depth_info,*rgb_info,*sub_rob_state) ;
            ksync->registerCallback(boost::bind(&SemMapper::kinectOdomCb, this, _1, _2, _3,_4,_5));
        }
        else
        {
            vsync =  new message_filters::Synchronizer<VeloOdom>(VeloOdom(10), *sub_velo_points,*sub_rob_state) ;
            vsync->registerCallback(boost::bind(&SemMapper::veloOdomCb, this, _1, _2));
        }


        Eigen::AngleAxisd Rx = Eigen::AngleAxisd(-M_PI/2,Eigen::Vector3d::UnitX());
        Eigen::AngleAxisd Rz = Eigen::AngleAxisd(-M_PI/2,Eigen::Vector3d::UnitZ());
        Eigen::Quaternion<double> R =  Rz*Rx;
        Tcam_to_sensor = Eigen::Matrix4d::Identity();
        Tcam_to_sensor.topLeftCorner(3,3)=R.matrix();
        Tcam_to_sensor.topRightCorner(3,1) = Eigen::Vector3d(0,0,-1.0);

//cout<<"Tcam_to_sensor*(1,2,3,1)="<<Tcam_to_sensor*Eigen::Vector4d(1,2,3,1)<<endl;
        Tsensor_to_rob = Eigen::Matrix4d::Identity();
        Eigen::AngleAxisd A = (Eigen::AngleAxisd(sensor_pose_t,Eigen::Vector3d::UnitZ()));
        R = A;
        Tsensor_to_rob.topLeftCorner(3,3) = R.matrix();
        Tsensor_to_rob.topRightCorner(3,1) = Eigen::Vector3d(sensor_pose_x,sensor_pose_y,sensor_pose_z);

//cout<<"Tsensor_to_rob*(1,2,3,1)="<<Tsensor_to_rob*Eigen::Vector4d(1,2,3,1)<<endl;
       // Viewing

        string models_path;
        nh.param("models_path",models_path,string(""));
        models.push_back(models_path+"pallets.svm");
        models.push_back(models_path+"pillar.svm");
        models.push_back(models_path+"depository.svm");
        ObjectDetector pd(models[0]);
        detectors.push_back(pd);
        ObjectDetector ld(models[1]);
        detectors.push_back(ld);
        ObjectDetector dd(models[2]);
        detectors.push_back(dd);

        if(visualize)
        {
            viewer = new SemViz(true);
            viewer->win3D->start_main_loop_own_thread();
        }
        isPoseInit =false;

        semEx->VISUALIZE = visualize;
        sem_saver = nh.advertiseService("save_semmap", &SemMapper::save_map_callback, this);


#ifdef WRITE_RUNTIME

        ofs.open("/home/sg/semMapPub/runtime.csv",std::ios::out);
#endif

        cuda::setDevice(0);
    }

    ~SemMapper(){
        delete viewer;
#ifdef WRITE_RUNTIME
        ofs.close();
#endif
                }

    void state_callback(const nav_msgs::Odometry::ConstPtr& msg_in)
    {
        Eigen::Quaterniond qd;
        Eigen::Affine3d gt_pose;
        float x,y,z;

        qd.x() = msg_in->pose.pose.orientation.x;
        qd.y() = msg_in->pose.pose.orientation.y;
        qd.z() = msg_in->pose.pose.orientation.z;
        qd.w() = msg_in->pose.pose.orientation.w;

        gt_pose = Eigen::Translation3d (msg_in->pose.pose.position.x,
                                        msg_in->pose.pose.position.y,msg_in->pose.pose.position.z) * qd;


        m.lock();

//        ROS_INFO("got GT pose from GT track (%f,%f,%f)",pose.translation()(0),pose.translation()(1),pose.translation()(2));
        pose =  gt_pose;

        x = pose.translation()(0);
        y = pose.translation()(1);
        z = pose.translation()(2);

        Trob_to_world = Eigen::Matrix4d::Identity();
        Trob_to_world.topLeftCorner(3,3) = pose.rotation();
        Trob_to_world.topRightCorner(3,1) = pose.translation();

        Tcam_to_world = Trob_to_world * Tsensor_to_rob * Tcam_to_sensor;


if(!isPoseInit)
{


    isPoseInit = true;
    if(visualize)
    {
        viewer->addTrajectoryPoint(x,y,z,1,0,0);
        viewer->displayTrajectory();
        viewer->setCameraPointing(x,y,z+3);
        viewer->win3D->setOrigin(x,y,z);
        viewer->win3D->draw_origin();
        viewer->repaint();
    }
    semMap->setInitPosition(x,y,z);

}
if(visualize)
{
    viewer->addTrajectoryPoint(x,y,z,1,0,0);
    viewer->displayTrajectory();
}
        m.unlock();



    }

    void veloOdomCb(const sensor_msgs::PointCloud2ConstPtr cloud,const nav_msgs::Odometry::ConstPtr state)
    {
        state_callback(state);

    }

    void processKinect(const sensor_msgs::ImageConstPtr& depth_msg,
                       const sensor_msgs::ImageConstPtr& rgb_msg_in,
                       const sensor_msgs::CameraInfoConstPtr& info_msg,cv::Mat &rgbImg,cv::Mat &depthImg)
    {
        // Check for bad inputs
        if (depth_msg->header.frame_id != rgb_msg_in->header.frame_id)
        {
          ROS_INFO("Depth image frame id [%s] doesn't match RGB image frame id [%s]",
                                 depth_msg->header.frame_id.c_str(), rgb_msg_in->header.frame_id.c_str());
          return;
        }

        // Update camera model
        model_.fromCameraInfo(info_msg);

        // Check if the input image has to be resized
        sensor_msgs::ImageConstPtr rgb_msg = rgb_msg_in;
        cv_bridge::CvImagePtr cv_ptr;
        try
        {
          cv_ptr = cv_bridge::toCvCopy(rgb_msg, rgb_msg->encoding);

        }
        catch (cv_bridge::Exception& e)
        {
          ROS_ERROR("cv_bridge exception: %s", e.what());
          return;
        }
        if (depth_msg->width != rgb_msg->width || depth_msg->height != rgb_msg->height)
        {
          sensor_msgs::CameraInfo info_msg_tmp = *info_msg;
          info_msg_tmp.width = depth_msg->width;
          info_msg_tmp.height = depth_msg->height;
          float ratio = float(depth_msg->width)/float(rgb_msg->width);
          info_msg_tmp.K[0] *= ratio;
          info_msg_tmp.K[2] *= ratio;
          info_msg_tmp.K[4] *= ratio;
          info_msg_tmp.K[5] *= ratio;
          info_msg_tmp.P[0] *= ratio;
          info_msg_tmp.P[2] *= ratio;
          info_msg_tmp.P[5] *= ratio;
          info_msg_tmp.P[6] *= ratio;
          model_.fromCameraInfo(info_msg_tmp);

          cv_bridge::CvImage cv_rsz;
          cv_rsz.header = cv_ptr->header;
          cv_rsz.encoding = cv_ptr->encoding;
          cv::resize(cv_ptr->image.rowRange(0,depth_msg->height/ratio), cv_rsz.image, cv::Size(depth_msg->width, depth_msg->height));
          rgb_msg = cv_rsz.toImageMsg();

          //ROS_INFO(5, "Depth resolution (%ux%u) does not match RGB resolution (%ux%u)",
          //                       depth_msg->width, depth_msg->height, rgb_msg->width, rgb_msg->height);
          //return;
        } else
          rgb_msg = rgb_msg_in;

        // Supported color encodings: RGB8, BGR8, MONO8
        int red_offset, green_offset, blue_offset, color_step;
        if (rgb_msg->encoding == enc::RGB8)
        {
          red_offset   = 0;
          green_offset = 1;
          blue_offset  = 2;
          color_step   = 3;
        }
        else if (rgb_msg->encoding == enc::BGR8)
        {
          red_offset   = 2;
          green_offset = 1;
          blue_offset  = 0;
          color_step   = 3;
        }
        else if (rgb_msg->encoding == enc::MONO8)
        {
          red_offset   = 0;
          green_offset = 0;
          blue_offset  = 0;
          color_step   = 1;
        }
        else
        {
          ROS_INFO("Unsupported encoding [%s]", rgb_msg->encoding.c_str());
          return;
        }

        rgbImg = cv_ptr->image;
        try
        {
          cv_ptr = cv_bridge::toCvCopy(depth_msg, depth_msg->encoding);

        }
        catch (cv_bridge::Exception& e)
        {
          ROS_ERROR("cv_bridge exception: %s", e.what());
          return;
        }
        depthImg = cv_ptr->image;
    }

    void kinectOdomCb(const sensor_msgs::ImageConstPtr& depth_msg,
                                          const sensor_msgs::ImageConstPtr& rgb_msg_in,
                                          const sensor_msgs::CameraInfoConstPtr& depth_info_msg,const sensor_msgs::CameraInfoConstPtr& rgb_info_msg,const nav_msgs::Odometry::ConstPtr state)
    {
        nframes++;
        state_callback(state);

//        ROS_INFO("Sensor Data Received");


      cv::Mat rgbImg,depthImg;

      processKinect(depth_msg,rgb_msg_in,depth_info_msg,rgbImg,depthImg);

      Eigen::Matrix3f K(Eigen::Matrix3f::Zero());
      K(0,0) = model_.fx();
      K(1,1) = model_.fy();
      K(0,2) = model_.cx();
      K(1,2) = model_.cy();

      ros::Time timer = ros::Time::now();

      double targetTime,salTime,updTime,totTime;

std::vector<SemRegion> R;

semEx->setImages(rgbImg,depthImg);
semEx->setCameraParams(K,Tcam_to_world);
      semEx->extractTargetRegions(detectors,R);

      targetTime = (ros::Time::now()-timer).toSec();
      ROS_INFO("Target Regions extraction time taken = %f",targetTime);

      timer = ros::Time::now();

      semEx->extractSalientRegions(R);

      salTime = (ros::Time::now()-timer).toSec();
      ROS_INFO("Salient Regions extracted time taken = %f",salTime);

      timer = ros::Time::now();
updTime = 0;
        semEx->updateMap(R);
        if(semMap->colors.size()<semMap->semRegions.size())
            getColors(semMap->colors,semMap->semRegions.size());


      semEx->toCloud(2);


if(mapRegions!=semMap->semRegions.size())
      ROS_INFO("Map has %d unique Regions...\n\n",semMap->semRegions.size());
mapRegions = semMap->semRegions.size();
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZRGB>);
        pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> outrem;
        outrem.setInputCloud(semEx->cloud);
        outrem.setKeepOrganized(true);
        outrem.setMeanK(5);
        outrem.setStddevMulThresh(0.01);
        outrem.filter (*cloud_filtered);
        semEx->cloud = cloud_filtered;
//ROS_INFO("Adding cloud with %d points",semEx->cloud->points.size());
        if(semEx->cloud->points.size()>10)
            semMap->occmap.addCloud(*semEx->cloud,Tcam_to_world);
//pcl::transformPointCloud(*semEx->cloud,*semEx->cloud,Tcam_to_world);

updTime = (ros::Time::now()-timer).toSec();

if(nframes%2==0 & visualize)
{
    //semEx->getMap(semMap);
      this->viewer->viewSemMap(semMap);
//    this->viewer->viewSemMapRegions(semMap);
    //pcl::transformPointCloud(*semEx->cloud,*semEx->cloud,Tcam_to_world);
    //semEx->showCloud();

ROS_INFO("Map updated time taken = %f",updTime);
}
totTime = updTime + targetTime + salTime;


#ifdef WRITE_RUNTIME
ofs << targetTime << "\t" << salTime << "\t" << updTime << "\t" << totTime <<endl;
#endif
    }

    bool save_map_callback(std_srvs::Empty::Request  &req,std_srvs::Empty::Response &res ) {

        bool status = false;
        ROS_INFO("Saving Semantic Map %s", mapPath.c_str());
        m.lock();
        status = semMap->saveMap(mapPath,"map");
        m.unlock();
        return status;
      }


};





int main(int argc, char **argv)
{
    const rlim_t stackSize = 100 * 1024 * 1024;
    struct rlimit rl;
    int result;

    result = getrlimit(RLIMIT_STACK, &rl);
    rl.rlim_cur = stackSize;
    result = setrlimit(RLIMIT_STACK,&rl);


  ros::init(argc, argv, "sem_map");

  ros::NodeHandle nh("~");
  //NDTMapper t(nh);
  SemMapper sm(nh);
  ros::spin();

  return 0;
}
