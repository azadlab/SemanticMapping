/*

Author Name: J. Rafid S.
Author URI: www.azaditech.com
Relevant Publication: Siddiqui, J. R. , Andreasson, H. , Driankov, D. & Lilienthal, A. J. (2016). Towards visual mapping in industrial environments: a heterogeneous task-specific and saliency driven approach. IEEE International Conference on Robotics and Automation (ICRA). , Stockholm, Sweden, May, 2016.

*/


#include <iostream>
#include <iomanip>
#include <semantic_mapping/ObjectDetector.h>
#include <dlib/array2d.h>
#include <dlib/data_io.h>
#include <dirent.h>
#include <sys/stat.h>
#include <dlib/opencv/cv_image.h>
#include <dlib/opencv.h>
#include <fstream>

struct stat filestat;
using namespace std;


float compute_TPR(dlib::array<dlib::array2d<unsigned char> > &images, std::vector<std::vector<dlib::rectangle> > &gt_rects,
                  ObjectDetector &detector)
{
    float tot_detections =0 , true_detections = 0;
    for(int i=0;i<images.size();i++)
    {

        Mat img;
        img = dlib::toMat(images[i]);
        resize(img,img,cv::Size(img.rows,img.rows));
        std::vector<dlib::rectangle> obj_gt = gt_rects[i];
        std::vector<dlib::rectangle> detect_rects = detector.detect(images[i]);

        for(int j=0;j<detect_rects.size();j++)
        {
            dlib::rectangle drect = detect_rects[j];
            for(int k=0;k<obj_gt.size();k++)
                if (drect.intersect(obj_gt[k]).area()>0.7*drect.area())
                    true_detections++;
            tot_detections++;
        }

    }

    return true_detections / tot_detections;
}


int main(int argc,char** argv)
{

    std::vector<ObjectDetector> detectors;
    string models_path("/media/sg/work/coop/object_detection/");


    ObjectDetector pd(models_path+"pallets.svm");
    detectors.push_back(pd);
    ObjectDetector ld(models_path+"pillar.svm");
    detectors.push_back(ld);
    ObjectDetector dd(models_path+"depository.svm");
    detectors.push_back(dd);
    string dataset_path(argv[1]);

    string pallets_gt_path = dataset_path + "/pallets_gt.xml";
    string pillar_gt_path = dataset_path + "/pillar_gt.xml";
    string depository_gt_path = dataset_path + "/depository_gt.xml";

    dlib::array<dlib::array2d<unsigned char> > images;
    std::vector<std::vector<dlib::rectangle> > object_rects;
    dlib::load_image_dataset(images,object_rects,pallets_gt_path);

    float pallet_tpr = compute_TPR(images,object_rects,detectors[0]);

    dlib::load_image_dataset(images,object_rects,pillar_gt_path);

    float pillar_tpr = compute_TPR(images,object_rects,detectors[1]);

    dlib::load_image_dataset(images,object_rects,depository_gt_path);

    float depository_tpr = compute_TPR(images,object_rects,detectors[2]);

    std::ofstream ofs;

    ofs.open(models_path+"/object_detector.result",std::ios::out);

    ofs << "Object\t" << "TPR" <<endl;
    ofs<< "Pallet\t" << pallet_tpr <<endl << "Pillar\t" << pillar_tpr <<endl << "Depository\t" << depository_tpr <<endl;

    ofs.close();

    return 0;
}
