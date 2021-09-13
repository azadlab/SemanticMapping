/*

Author Name: J. Rafid S.
Author URI: www.azaditech.com
Relevant Publication: Siddiqui, J. R. , Andreasson, H. , Driankov, D. & Lilienthal, A. J. (2016). Towards visual mapping in industrial environments: a heterogeneous task-specific and saliency driven approach. IEEE International Conference on Robotics and Automation (ICRA). , Stockholm, Sweden, May, 2016.

*/


#include <iostream>
#include <iomanip>
#include <semantic_mapping/SaliencyExtractor.h>
#include <dirent.h>
#include <sys/stat.h>
#include <dlib/opencv/cv_image.h>
#include <dlib/opencv.h>
#include <dlib/data_io.h>

struct stat filestat;
using namespace std;

#define DEMO

vector<float> compute_TPR(dlib::array<dlib::array2d<dlib::rgb_pixel> > &images, vector<std::vector<dlib::rectangle>> &gt_rects)
{

    vector<float> tpr;
    for(int i=0;i<images.size();i++)
    {

        Mat img;

        img = dlib::toMat(images[i]);
        cvtColor(img,img,CV_RGB2BGR);
        resize(img,img,cv::Size(img.rows,img.rows));

        SaliencyExtractor salEx(img.rows*0.04,AttentionModel::SCAN_FIX);
        salEx.compute(img);
        Mat salmap;
        salmap = salEx.getSaliencyMap();
        //salmap.convertTo(salmap,CV_8UC1,255);
//        imshow("IMG",img);
//        imshow("salmap",salmap);waitKey(0);
        vector<dlib::rectangle> rects = gt_rects[i];

        for(int k=0;k<rects.size();k++)
        {
            dlib::rectangle grect = rects[k];
            if(grect.area() > 1000)
            {
            long x = max((long)0,grect.left());
            long y = max((long)0,grect.top());
            long width = grect.width();
            long height = grect.height();
            width = x+width<img.cols?width:img.cols-x;
            height = y+height<img.rows?height:img.rows-y;
            Mat salRegion(salmap(Rect(x,y,width,height)));
            Mat salFlag = salRegion > mean(salmap)[0];
            salFlag.convertTo(salFlag,CV_32FC1,1/255.0);
            float tot_area = salFlag.rows*salFlag.cols;
            float sal_area = sum(salFlag)[0];
            float rtpr = sal_area/tot_area;

            tpr.push_back(rtpr);
            }
        }

    }

    return tpr;
}


int demo_saliency(char** argv)
{
    string dataset_path(argv[1]);
    string output_path(argv[2]);

    int start = atoi(argv[3]);
    int end = atoi(argv[4]);

    vector<string> files;
    ostringstream oss;

    DIR *dir;
    struct dirent *ent;
    if ((dir = opendir (dataset_path.c_str())) != NULL) {
      while ((ent = readdir (dir)) != NULL) {
          char filename[512];
          snprintf(filename, sizeof(filename), "%s/%s", dataset_path.c_str(), ent->d_name);
          lstat(filename, &filestat);

          if (S_ISREG( filestat.st_mode ))
              files.push_back(ent->d_name);
      }
      closedir (dir);
    } else {

      perror ("");
      return EXIT_FAILURE;
    }

    vector<Mat> OCS;
    for(int i=start;i<end;i++)
    {

        //oss << dataset_path << "/images/image" << std::setfill ('0') << std::setw (3) << i << ".jpg";
//        oss << dataset_path << "/images/" << std::setfill ('0') << std::setw (4) << i << ".jpg";
        oss << dataset_path << "/" <<files[i];
        cout<<"Processing file:"<<oss.str()<<endl;
        Mat img,segImg;
        img = imread(oss.str());
        resize(img,img,cv::Size(img.rows,img.rows));
        //imshow("Image",img);


        SaliencyExtractor salEx(img.rows*0.05,AttentionModel::SCAN_FIX);
        salEx.compute(img);
        Mat salmap,salmask,Gsal,Lsal;
        salmap = salEx.getSaliencyMap();
        salmask = salEx.getSaliencyMask();
        Gsal = salEx.getGlobalSaliencyMap();
        Lsal = salEx.getLocalSaliencyMap();
        segImg = salEx.getSegmentImage();
        OCS.clear();
        OCS = salEx.getOpponentSpace();
        cv::Mat mimg;
        img.copyTo(mimg,salmask);
        imshow("salmask",mimg);
        salmap.convertTo(salmap,CV_8UC1,255);
        Gsal.convertTo(Gsal,CV_8UC1,255);
        Lsal.convertTo(Lsal,CV_8UC1,255);

        oss.str("");
        oss << output_path << "/saliency/joint/" <<files[i];
        imwrite(oss.str(),salmap);
        oss.str("");
        oss << output_path << "/saliency/global/" <<files[i];
        imwrite(oss.str(),Gsal);
        oss.str("");
        oss << output_path << "/saliency/local/" <<files[i];
        imwrite(oss.str(),Lsal);
        oss.str("");
        oss << output_path << "/saliency/mask/" <<files[i];
        imwrite(oss.str(),mimg);
        oss.str("");
        oss << output_path << "/saliency/segmentation/" <<files[i];
        imwrite(oss.str(),segImg);

        Mat C1,C2,C3,C4;
        normalize(OCS[0],C1,0,255,NORM_MINMAX,CV_8UC1);
        normalize(OCS[1],C2,0,255,NORM_MINMAX,CV_8UC1);
        normalize(OCS[2],C3,0,255,NORM_MINMAX,CV_8UC1);
        normalize(OCS[3],C4,0,255,NORM_MINMAX,CV_8UC1);
        imshow("O1",C1);
        imshow("O2",C2);
        imshow("O3",C3);
        imshow("O4",C4);


        oss.str("");
        oss << output_path << "/saliency/color_space/" <<"_O1" <<files[i];
        imwrite(oss.str(),C1);
        oss.str("");
        oss << output_path << "/saliency/color_space/" <<"_O2" <<files[i];
        imwrite(oss.str(),C2);
        oss.str("");
        oss << output_path << "/saliency/color_space/" <<"_O3" <<files[i];
        imwrite(oss.str(),C3);
        oss.str("");
        oss << output_path << "/saliency/color_space/" <<"_O4" <<files[i];
        imwrite(oss.str(),C4);

//        cv::waitKey(0);

        oss.str("");
    }
}

int main(int argc,char** argv)
{

#ifdef DEMO
    demo_saliency(argv);


#else

    string dataset_path(argv[1]);
string gt_path = dataset_path + "/saliency/saliency.xml";
dlib::array<dlib::array2d<dlib::rgb_pixel> > images;
std::vector<std::vector<dlib::rectangle> > object_rects;
dlib::load_image_dataset(images,object_rects,gt_path);
vector<float> tpr = compute_TPR(images,object_rects);


std::ofstream ofs;

ofs.open(dataset_path+"/saliency/result.csv",std::ios::out);

ofs << "Images\t" << "TPR" <<endl;
for(int i=0;i<tpr.size();i++)
    ofs<< i+1 << "\t" << tpr[i] <<endl;

ofs.close();

#endif

    return 0;
}
