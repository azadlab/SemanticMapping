/*

Class Name: ObjectDetector
Author Name: J. Rafid S.
Author URI: www.azaditech.com
Description: Accepts a string of model name and generates a class object for detecting objects.
Relevant Publication: Siddiqui, J. R. , Andreasson, H. , Driankov, D. & Lilienthal, A. J. (2016). Towards visual mapping in industrial environments: a heterogeneous task-specific and saliency driven approach. IEEE International Conference on Robotics and Automation (ICRA). , Stockholm, Sweden, May, 2016.

*/

#ifndef OBJECTDETECTOR_H
#define OBJECTDETECTOR_H

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <dlib/svm_threaded.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_processing.h>
#include <dlib/opencv/cv_image.h>
#include <dlib/data_io.h>
#include <time.h>

#define OBJECT_LIFE 1

using namespace std;
using namespace cv;



typedef dlib::scan_fhog_pyramid<dlib::pyramid_down<6> > image_scanner_type;


struct Tracker
{
    dlib::correlation_tracker ct;
    clock_t birth;
};

class ObjectDetector
{

dlib::object_detector<image_scanner_type> detector;
std::vector<dlib::object_detector<image_scanner_type> > detectors;
std::vector<Tracker> trackers;
bool MultiDetection;

public:


    ObjectDetector(string model)
    {
        dlib::deserialize(model)>>detector;
        MultiDetection = false;

    }
    ObjectDetector(std::vector<string> models)
    {
        MultiDetection = true;
        for(int i=0;i<models.size();i++)
        {
            dlib::deserialize(models[i])>>detector;
            detectors.push_back(detector);
        }

    }

std::vector<dlib::rectangle> detect(Mat &img);
std::vector<dlib::rectangle> detect(dlib::array2d<unsigned char> &img);
void markImage(Mat&,std::vector<dlib::rectangle>&,cv::Scalar);
void addtracker(dlib::cv_image<dlib::bgr_pixel> &img, std::vector<dlib::rectangle> &);
void updatetracker(dlib::cv_image<dlib::bgr_pixel> &img, std::vector<dlib::rectangle> &);
dlib::matrix<unsigned char> getFeatures() {return draw_fhog(detector);}
};

#endif
