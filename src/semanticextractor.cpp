/*

Author Name: J. Rafid S.
Author URI: www.azaditech.com
Relevant Publication: Siddiqui, J. R. , Andreasson, H. , Driankov, D. & Lilienthal, A. J. (2016). Towards visual mapping in industrial environments: a heterogeneous task-specific and saliency driven approach. IEEE International Conference on Robotics and Automation (ICRA). , Stockholm, Sweden, May, 2016.

*/



#include "semantic_mapping/semanticextractor.h"
#include <pcl/search/organized.h>

const float dsampfact = 1.875;
const int Xcrop = 42;
const int Ycrop = 0;
const float DLimit = 10.0;



void semanticExtractor::filterDepth(cv::Mat src, cv::Mat &des,int wsize,int numIter,float thresh)
{


    cv::Mat varMat,mu;
    for(int i=0;i<numIter;i++)
    {
        //cv::bilateralFilter(src,mu,wsize,0,wsize);
        cv::blur(src,mu,cv::Size(wsize,wsize));
        varMat = src-mu;
        varMat=varMat.mul(varMat);
        src.setTo(cv::Scalar(0),(varMat>thresh));

    }

    des = src;
}

void semanticExtractor::extractTargetRegions(std::vector<ObjectDetector> &detectors,std::vector<SemRegion>& Regions)
{


    cv::Mat auxImg;
    //cv::resize(rgbImg,auxImg,cv::Size(rgbImg.cols/dsampfact/2,rgbImg.rows/dsampfact/2));
    cv::resize(rgbImg,auxImg,cv::Size(rgbImg.cols/dsampfact,rgbImg.rows/dsampfact));
//    rgbImg = colorReduce(rgbImg,16);

    cv::Rect ROI(Xcrop,Ycrop,auxImg.cols-Xcrop*2-1,auxImg.rows-Ycrop);
    rgbImg = auxImg(ROI);
//    cv::medianBlur(rgbImg,rgbImg,Smoothness);
cv::bilateralFilter(rgbImg,auxImg,5,200,20);
rgbImg = auxImg;

    cv::resize(depthImg,auxImg,cv::Size(depthImg.cols/dsampfact/2,depthImg.rows/dsampfact/2));
    //cv::GaussianBlur(auxImg,auxImg,cv::Size(Smoothness,Smoothness),0,0);
    cv::resize(auxImg,depthImg,cv::Size(depthImg.cols/dsampfact,depthImg.rows/dsampfact));


auxImg = cv::Mat::zeros(depthImg.size(), depthImg.type());
depthImg(cv::Rect(0,0,depthImg.cols-5,depthImg.rows-4)).copyTo(auxImg(cv::Rect(5,4,depthImg.cols-5,depthImg.rows-4)));
depthImg = auxImg;
    depthImg = depthImg(ROI);
depthImg(cv::Rect(0,0,depthImg.cols,70)).setTo(cv::Scalar(0.));
    double minval,maxval;

depthImg.setTo(cv::Scalar(0),depthImg>DLimit*1000);
depthImg.convertTo(auxImg,CV_32F,1/1000.0);

depthImg = cv::Mat(auxImg.size(),auxImg.type());
filterDepth(auxImg,depthImg,11,3,0.3);
auxImg = depthImg;


auxImg.convertTo(depthImg, CV_8UC1, 255/DLimit);


cv::cvtColor(rgbImg,labImg,CV_BGR2Lab);

mask = cv::Mat(rgbImg.size(),rgbImg.type(),cv::Scalar(0,0,255));

    std::vector<dlib::rectangle> prects = detectors[0].detect(rgbImg);
    std::vector<dlib::rectangle> lrects = detectors[1].detect(rgbImg);
    std::vector<dlib::rectangle> drects = detectors[2].detect(rgbImg);


    if(prects.size()>=1)
    {
        SemRegion R = extractSemRegion(prects,Pallet,cv::Vec3f(1,1,0));
        if(R.isvalid)
            Regions.push_back(R);
        ObjectsDetected = true;
    }
    if(lrects.size()>=1)
    {
        SemRegion R = extractSemRegion(lrects,Pillar,cv::Vec3f(0,0,0));
        if(R.isvalid)
            Regions.push_back(R);
        ObjectsDetected = true;
    }
    if(drects.size()>=1)
    {
        SemRegion R = extractSemRegion(drects,Depository,cv::Vec3f(0,1,1));
        if(R.isvalid)
            Regions.push_back(R);
        ObjectsDetected = true;
    }


    if(semanticExtractor::VISUALIZE)
    {

        //cv::convertScaleAbs(depthImg,auxImg,255/1000.0f);
        cv::imshow(DEPTH_WINDOW,depthImg);
        cv::imshow(RGB_WINDOW,rgbImg);
        cv::imshow(SEG_WINDOW,mask);

        cv::waitKey(1);
    }



}


void semanticExtractor::extractSalientRegions(std::vector<SemRegion> &regions)
{
    if(!ObjectsDetected)
    {
        cv::Mat salmask(depthImg.size(),depthImg.type(),cv::Scalar(0.));

        SaliencyExtractor salEx(4,AttentionModel::RAND_FIX);
        cv::Mat auxImg(depthImg.size(),rgbImg.type(),cv::Scalar(0.));

        cv::resize(rgbImg,auxImg,rgbImg.size()/2);

        salEx.compute(auxImg);
        auxImg = salEx.getSaliencyMask();

        cv::resize(auxImg,salmask,rgbImg.size());
        mask = salmask;
        vector<SceneSegment> salSegs = salEx.getSalientRegions();
        for(int i=0;i<salSegs.size();i++)
        {
            SceneSegment S = salSegs[i];

            SemRegion R = extractSemRegion(S.bbox,Unknown,cv::Vec3f(S.color[2],S.color[1],S.color[0]));
            if(R.getSemLabel()==Unknown)
                regions.push_back(R);
        }

        if(semanticExtractor::VISUALIZE)
        {
            imshow(SAL_WINDOW,salmask);
        }
    }
    ObjectsDetected = false;
}

std::vector<cv::Mat> semanticExtractor::extractOriHist(std::vector<dlib::rectangle>& rects,SemLabel label)
{

    std::vector<cv::Mat> feats;
    vector<dlib::rectangle> filt_rects;

    for(int i=0;i<rects.size();i++)
    {
        dlib::array2d<dlib::matrix<float,31,1> > hog;
        long x = rects[i].left();
        long y = rects[i].top();
        x = x<0?0:x;
        y = y<0?0:y;
        long width = rects[i].width();
        long height = rects[i].height();
        width = (width+x > rgbImg.cols)?(rgbImg.cols-x):width;
        height = (height+y > rgbImg.rows)?(rgbImg.rows-y):height;
        if(x<0|y<0|width<0|height<0) continue;
        Rect roi(x,y,width,height);

        Mat img;
        try{

            img = rgbImg(roi);

            if(label==Pallet)
                cv::rectangle(mask,roi,cv::Scalar(0,255,255),CV_FILLED);
            else if(label == Pillar)
                cv::rectangle(mask,roi,cv::Scalar(0,0,0),CV_FILLED);
            else if(label == Depository)
                cv::rectangle(mask,roi,cv::Scalar(0,255,0),CV_FILLED);

        }catch(cv::Exception ex){cout<<"Exception while extracting HOG Regions"<<endl;
                                 cout<<"ROI:"<<roi<<endl;
                                 cout<<"Image Size:"<<rgbImg.rows<<", "<<rgbImg.cols<<endl;
                                 exit(-1);}
        dlib::cv_image<dlib::bgr_pixel> cvimg(img);

        dlib::extract_fhog_features(cvimg, hog);

        cv::Mat aux(cv::Mat::zeros(1,31,CV_32F));

        for(int r=0;r<hog.nr();r++)
            for(int c=0;c<hog.nc();c++)
            {
                float* auxPtr = aux.ptr<float>(0);
                for(int j=0;j<31;j++)
                {
                    auxPtr[j] += hog[r][c](j);
                }
            }
        if(hog.nr()*hog.nc()>0)
        {
            aux /= hog.nr()*hog.nc();
            filt_rects.push_back(rects[i]);
            feats.push_back(aux);
        }
    }

        rects = filt_rects;
        return feats;
}

std::vector<cv::Mat> semanticExtractor::extractColors(std::vector<dlib::rectangle>& rects)
{
    std::vector<cv::Mat> colors;
    for(int i=0;i<rects.size();i++)
    {
        int x = rects[i].left();
        int y = rects[i].top();
        x = x<0?0:x;
        y = y<0?0:y;
        int width = rects[i].width();
        int height = rects[i].height();
        width = (width+x > labImg.cols)?(labImg.cols-x):width;
        height = (height+y > labImg.rows)?(labImg.rows-y):height;
        if(x<0|y<0|width<0|height<0) continue;
        cv::Mat auxMat;
        try{
            Rect roi(x,y,width,height);
            auxMat = rgbImg(roi);


        }catch(cv::Exception ex){cout<<"Exception while Color Region Extraction"<<endl;exit(-1);}
        cv::Scalar v = cv::mean(auxMat);
        auxMat.setTo(cv::Scalar(0,0,0));
        cv::Mat c(1,3,CV_8U);
        c.at<uchar>(0)= v[0];
        c.at<uchar>(1)= v[1];
        c.at<uchar>(2)= v[2];
        colors.push_back(c);
    }
    return colors;
}




void semanticExtractor::toCloud(int step)
{


    float cx = (K(0,2)/dsampfact)-Xcrop;
    float cy = (K(1,2)/dsampfact)-Ycrop;
    this->cloud->clear();

    for (int i = 0; i < mask.rows; i+=step) {
        for (int j = 0; j < mask.cols; j+=step) {

            uchar B = mask.at<cv::Vec3b>(i, j)[0];
            uchar G = mask.at<cv::Vec3b>(i, j)[1];
            uchar R = mask.at<cv::Vec3b>(i, j)[2];

                pcl::PointXYZRGB pt;
                pt.z = (float)depthImg.at<uchar>(i,j)*(DLimit/255);
                if(pt.z==pt.z && pt.z>1.0)
                {
                    pt.x = (j-cx)*dsampfact*(1/K(0,0))*pt.z;
                    pt.y = (i-cy)*dsampfact*(1/K(1,1))*pt.z;


                    pt.r = R;
                    pt.g = G;
                    pt.b = B;
                }
                else
                {


                    pt.x = std::numeric_limits<float>::quiet_NaN();
                    pt.y = std::numeric_limits<float>::quiet_NaN();
                    pt.z = std::numeric_limits<float>::quiet_NaN();
                    pt.r = std::numeric_limits<float>::quiet_NaN();
                    pt.g = std::numeric_limits<float>::quiet_NaN();
                    pt.b = std::numeric_limits<float>::quiet_NaN();

                }


                this->cloud->push_back(pt);

        }
    }
    cloud->width = mask.cols/step;
    cloud->height = mask.rows/step;
    cloud->is_dense = true;
}




Eigen::Matrix4d semanticExtractor::computePose(Eigen::Vector3d pixPos,Eigen::Matrix4d cam_to_world)
{
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();

//    Eigen::Vector3d mapPos = robPose * Eigen::Vector3d(-pixPos(0),-pixPos(2),-pixPos(1));
    Eigen::Vector4d mapPos = Eigen::Vector4d::Zero();
    mapPos(0) = pixPos(0);mapPos(1) = pixPos(1);mapPos(2) = pixPos(2);
    mapPos(3) = 1;
    mapPos = cam_to_world * mapPos;
    mapPos /= mapPos(3);
    T(0,3) = mapPos(0);
    T(1,3) = mapPos(1);
    T(2,3) = mapPos(2);

    return T;
}


SemRegion semanticExtractor::extractSemRegion(std::vector<dlib::rectangle>& rects,SemLabel label,cv::Vec3f c)
{

    SemRegion R;
    R.setSemLabel(label);

    std::vector<cv::Mat> X;
    std::vector<cv::Mat> Y;
    X = extractOriHist(rects,label);
    Y = extractColors(rects);

    if(X.size()>0 & Y.size()>0)
    for(int i=0;i<rects.size();i++)
    {

        Appearance a;
        a.Orientation = X[i];
        a.meanColor = Y[i];
        R.addAppearance(a);

        Eigen::Matrix4d T;
        int x = rects[i].left();
        int y = rects[i].top();
        x = x<0?0:x;
        y = y<0?0:y;
        int width = rects[i].width();
        int height = rects[i].height();
        width = (width+x > rgbImg.cols)?(rgbImg.cols-x):width;
        height = (height+y > rgbImg.rows)?(rgbImg.rows-y):height;
        if(x<0|y<0|width<0|height<0) continue;
        Mat img;
        Rect roi(x,y,width,height);
        img = depthImg(roi);
        float Z = cv::mean(img,img>0)[0]*(DLimit/255);
        int midx = (x + width)/2;
        int midy = (y + height)/2;

        float cx = K(0,2)/dsampfact-Xcrop;
        float cy = K(1,2)/dsampfact-Ycrop;
        float X = (midx-cx)*dsampfact*(1/K(0,0))*Z;
        float Y = (midy-cy)*dsampfact*(1/K(1,1))*Z;
//        cout<<"(x,y)=("<<midx<<", "<<midy<<endl;
//        cout<<"(X,Y,Z)=("<<X<<", "<<Y<<", "<<Z<<endl;
        T = computePose(Eigen::Vector3d(X,Y,Z),cam_to_world);
        cv::Mat pos(1,3,CV_32F);
        pos.at<float>(0)=T(0,3);
        pos.at<float>(1)=T(1,3);
        pos.at<float>(2)=T(2,3);
        R.addPose(pos);

    }
    else R.isvalid=false;
    return R;
}


SemRegion semanticExtractor::extractSemRegion(Rect& rect,SemLabel label,cv::Vec3f c)
{

    SemRegion R;
    R.setSemLabel(label);

   // std::vector<Eigen::VectorXd> O;
    std::vector<Eigen::Vector3d> C;
   // O = extractOriHist(rects,label);
   // C = extractColors(rect);

        Appearance a;
        //a.Orientation = O[i];
        a.meanColor = c;
        R.addAppearance(a);

        Eigen::Matrix4d T;
        int x = rect.x;
        int y = rect.y;
        x = x<0?0:x;
        y = y<0?0:y;
        int width = rect.width;
        int height = rect.height;
        width = (width+x > rgbImg.cols)?(rgbImg.cols-x):width;
        height = (height+y > rgbImg.rows)?(rgbImg.rows-y):height;
        if(x<0|y<0|width<0|height<0) {R.isvalid=false;return R;};
        Mat img;
        Rect roi(x,y,width,height);
        img = depthImg(roi);
        float Z = cv::mean(img,img>0)[0]*(DLimit/255);
        int midx = (x + width)/2;
        int midy = (y + height)/2;

        float cx = K(0,2)/dsampfact-Xcrop;
        float cy = K(1,2)/dsampfact-Ycrop;
        float X = (midx-cx)*dsampfact*(1/K(0,0))*Z;
        float Y = (midy-cy)*dsampfact*(1/K(1,1))*Z;
//        cout<<"(x,y)=("<<midx<<", "<<midy<<endl;
//        cout<<"(X,Y,Z)=("<<X<<", "<<Y<<", "<<Z<<endl;
        T = computePose(Eigen::Vector3d(X,Y,Z),cam_to_world);
        cv::Mat pos(1,3,CV_32F);
        pos.at<float>(0)=T(0,3);
        pos.at<float>(1)=T(1,3);
        pos.at<float>(2)=T(2,3);
        R.addPose(pos);


    return R;
}


float semanticExtractor::getDissimilarity(Appearance a1,Appearance a2)
{

}


float semanticExtractor::getDissimilarity(std::vector<Appearance> apps,Appearance a)
{
    float cost = 0;
    for(int i=0;i<apps.size();i++)
        cost += getDissimilarity(apps[i],a);
    return cost/apps.size();
}

double semanticExtractor::getDistToRegion(std::vector<Eigen::Matrix4d> poses,Eigen::Vector3d pixPos,Eigen::Matrix4d world_trans)
{
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    Eigen::Vector4d mapPos = Eigen::Vector4d::Zero();
    mapPos(0) = pixPos(0);mapPos(1) = pixPos(1);mapPos(2) = pixPos(2);
    mapPos(3) = 1;
    mapPos = world_trans * mapPos;
    double mindist = INFINITY;
    for(int i=0;i<poses.size();i++)
    {
        Eigen::Matrix4d pose = poses[i];
        float dist = sqrt( (pose(0,3)-mapPos(0))*(pose(0,3)-mapPos(0)) + (pose(1,3)-mapPos(1))*(pose(1,3)-mapPos(1)) + (pose(2,3)-mapPos(2))*(pose(2,3)-mapPos(2)) );
        if(mindist>dist)
            mindist=dist;


    }
    return mindist;
}

void semanticExtractor::updateMap(std::vector<SemRegion> &regions)
{
    for(int i=0;i<regions.size();i++)
    {
        SemRegion R = regions[i];
        if(semMap->semRegions.size()<1)
        {
           semMap->semRegions.push_back(new SemRegion(R));
           continue;
        }
        else
        {
              for(int j=0;j<semMap->semRegions.size();j++)
              {
                  SemRegion *Rm = &semMap->semRegions[j];
                  if(Rm->getSemLabel()==R.getSemLabel())
                  {
                      Rm->addAppearances(R.getAppearances());
                      Rm->addPoses(R.getMapOccurances());
                      break;
                  }
                  else if(j+1 == semMap->semRegions.size())
                  {
                      semMap->semRegions.push_back(new SemRegion(R));
                      break;
                  }
              }
        }

    }
}
