
/*

Class Name: SemanticExtractor
Author Name: J. Rafid S.
Author URI: www.azaditech.com
Description: Class for creating semantic extractor Object.
Relevant Publication: Siddiqui, J. R. , Andreasson, H. , Driankov, D. & Lilienthal, A. J. (2016). Towards visual mapping in industrial environments: a heterogeneous task-specific and saliency driven approach. IEEE International Conference on Robotics and Automation (ICRA). , Stockholm, Sweden, May, 2016.

*/

#ifndef SEMANTICEXTRACTOR_H
#define SEMANTICEXTRACTOR_H

#include<iostream>
#include<pcl/point_types.h>
#include <pcl/point_cloud.h>
#include<pcl/io/pcd_io.h>
#include <pcl/features/feature.h>
#include<pcl/filters/extract_indices.h>
#include<pcl/segmentation/progressive_morphological_filter.h>
#include <ros/ros.h>
//#include <pcl/segmentation/conditional_euclidean_clustering.h>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <pcl/features/integral_image_normal.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/io/png_io.h>
#include <semantic_mapping/semmap.h>
#include <semantic_mapping/utils.h>
#include <semantic_mapping/ObjectDetector.h>
#include <semantic_mapping/SaliencyExtractor.h>
#include <opencv2/cudaimgproc.hpp>



std::string DEPTH_WINDOW="Depth View";
std::string RGB_WINDOW="RGB View";
std::string SEG_WINDOW="Target Detection";
std::string SAL_WINDOW="Saliency Mask";
#define Smoothness 9


using namespace std;
using namespace Eigen;

typedef std::vector < pcl::PointCloud<pcl::PointXYZRGB>::Ptr, Eigen::aligned_allocator <pcl::PointCloud <pcl::PointXYZRGB>::Ptr > > PlaceSegments;


class semanticExtractor
{

    MatrixXf data;
    boost::mutex mtx;
    pcl::visualization::CloudViewer *viewer_cloud;
    cv::Mat rgbImg,depthImg,labImg,mask;
    Eigen::Matrix3f K;
    Eigen::Matrix4d cam_to_world;
    bool ObjectsDetected = false;

public:

    SemMapPtr semMap;
//    MRF* mrf;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud;

    semanticExtractor(SemMapPtr &sm)
    {
        this->semMap = sm;
        if(this->VISUALIZE)
        {
            cv::namedWindow(DEPTH_WINDOW);
            cv::moveWindow(DEPTH_WINDOW,10,270);
            cv::namedWindow(RGB_WINDOW);
            cv::moveWindow(RGB_WINDOW,10,10);
            cv::namedWindow(SEG_WINDOW);
            cv::moveWindow(SEG_WINDOW,330,10);
            cv::namedWindow(SAL_WINDOW);
            cv::moveWindow(SAL_WINDOW,330,270);
            //viewer_cloud = new pcl::visualization::CloudViewer("Point View");

        //read_binary<MatrixXd>("/home/sg/dictionary.dat",semMap->Dict);
        //cout<<"size of Dict="<<semMap->Dict.rows()<<","<<semMap->Dict.cols()<<endl;
        }
        cloud = boost::make_shared<pcl::PointCloud<pcl::PointXYZRGB> > ();
    }

    ~semanticExtractor()
    {
        if(this->VISUALIZE)
        {
            cv::destroyWindow(DEPTH_WINDOW);
            cv::destroyWindow(RGB_WINDOW);
            cv::destroyWindow(SEG_WINDOW);
            cv::destroyWindow(SAL_WINDOW);

        }
    }

    bool VISUALIZE;

    void inline addRegion(SemRegion *R) {semMap->semRegions.push_back(R);}
    void toCloud(int step);
    void extractSurfaceInfo(std::vector<float> &desc);
    void extractTargetRegions(std::vector<ObjectDetector>&,std::vector<SemRegion>&);
    void extractSalientRegions(std::vector<SemRegion>&);
    float getDissimilarity(Appearance a1,Appearance a2);
    float getDissimilarity(std::vector<Appearance> apps,Appearance a);
    void inline setCloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cptr) {cloud=cptr;}
    void inline getMap(SemMap* sem) {*sem = *semMap;}
    SemRegion extractSemRegion(std::vector<dlib::rectangle>&,SemLabel,cv::Vec3f);
    SemRegion extractSemRegion(Rect&,SemLabel,cv::Vec3f);
    void updateMap(std::vector<SemRegion> &);
    Eigen::Matrix4d computePose(Eigen::Vector3d,Eigen::Matrix4d);
    void inline clearMap() {semMap->semRegions.clear();}
    void inline showCloud() {viewer_cloud->showCloud(cloud);}
    double getDistToRegion(std::vector<Eigen::Matrix4d>,Eigen::Vector3d,Eigen::Matrix4d);
    void computeCoherencies(cv::Mat&);
    Eigen::VectorXd encodeFeature(Eigen::VectorXd X);
    void filterDepth(cv::Mat src, cv::Mat &des,int wsize,int numIter,float thresh);
    std::vector<cv::Mat> extractOriHist(std::vector<dlib::rectangle>&,SemLabel);
    std::vector<cv::Mat> extractColors(std::vector<dlib::rectangle>&);

    void inline setImages(cv::Mat&rimg,cv::Mat&dimg) {this->rgbImg=rimg;this->depthImg=dimg;}
    void inline setCameraParams(Eigen::Matrix3f intrinsics,Eigen::Matrix4d extrinsics) {this->K=intrinsics;this->cam_to_world=extrinsics;}
    cv::Mat inline getMask(){return this->mask;}
};

#endif
