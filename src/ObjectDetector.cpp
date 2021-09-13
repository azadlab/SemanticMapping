/*

Author Name: J. Rafid S.
Author URI: www.azaditech.com
Relevant Publication: Siddiqui, J. R. , Andreasson, H. , Driankov, D. & Lilienthal, A. J. (2016). Towards visual mapping in industrial environments: a heterogeneous task-specific and saliency driven approach. IEEE International Conference on Robotics and Automation (ICRA). , Stockholm, Sweden, May, 2016.

*/


#include <semantic_mapping/ObjectDetector.h>

std::vector<dlib::rectangle> ObjectDetector::detect(dlib::array2d<unsigned char> &img)
{
    std::vector<dlib::rectangle> rects;
    if(MultiDetection)
        rects = dlib::evaluate_detectors(detectors,img);
    else
      rects = detector(img);
    return rects;

}

std::vector<dlib::rectangle> ObjectDetector::detect(Mat &img)
{
    dlib::cv_image<dlib::bgr_pixel> cvimg(img);
    std::vector<dlib::rectangle> rects;

    if(MultiDetection)
        rects = dlib::evaluate_detectors(detectors,cvimg);
    else
        rects = detector(cvimg);

    //updatetracker(cvimg,rects);
    return rects;
}

void ObjectDetector::markImage(Mat & img,std::vector<dlib::rectangle> &rects,cv::Scalar color)
{
    for(int i=0;i<rects.size();i++)
        cv::rectangle(img,cv::Rect(rects[i].left(),rects[i].top(),rects[i].width(),rects[i].height()),color,3);

}


void ObjectDetector::addtracker(dlib::cv_image<dlib::bgr_pixel> &img,std::vector<dlib::rectangle> & rects)
{

    for(int i=0;i<rects.size();i++)
    {
        dlib::rectangle rect = rects[i];
        dlib::correlation_tracker ct;
        ct.start_track(img,rect);
        Tracker tracker;
        tracker.ct = ct;
        tracker.birth = clock();
        trackers.push_back(tracker);

    }
}


void ObjectDetector::updatetracker(dlib::cv_image<dlib::bgr_pixel> &img, std::vector<dlib::rectangle> & rects)
{

    std::vector<Tracker> auxTrackers;
    std::vector<dlib::rectangle> curr_rects;
    for(int i=0;i<trackers.size();i++)
    {
        Tracker tracker = trackers[i];
        double age = (double)(clock()-tracker.birth)/CLOCKS_PER_SEC;
        if(age<=OBJECT_LIFE)
        {
            tracker.ct.update(img);
            curr_rects.push_back(tracker.ct.get_position());
            auxTrackers.push_back(tracker);

        }
    }
    trackers = auxTrackers;

    addtracker(img,rects);    //Add current Objects

    curr_rects.insert(curr_rects.end(),rects.begin(),rects.end());
    rects = curr_rects;

}
