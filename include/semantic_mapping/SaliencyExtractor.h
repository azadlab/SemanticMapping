/*

Class Name: SaliencyExtractor
Author Name: J. Rafid S.
Author URI: www.azaditech.com
Description: Class for creating objects for saliency extraction.
Relevant Publication: Siddiqui, J. R. , Andreasson, H. , Driankov, D. & Lilienthal, A. J. (2016). Towards visual mapping in industrial environments: a heterogeneous task-specific and saliency driven approach. IEEE International Conference on Robotics and Automation (ICRA). , Stockholm, Sweden, May, 2016.

*/

#ifndef SALIENTEXTRACTOR_H
#define SALIENTEXTRACTOR_H

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <semantic_mapping/utils.h>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaarithm.hpp>

using namespace std;
using namespace cv;

#define SEG_MIN 10
#define MIN_AREA 100
#define BINS 8
#define BG_THRESH 0.2
#define NUM_NHBS 10

enum class AttentionModel  {RAND_FIX, SCAN_FIX, CENTER_BIAS};
enum Stimulus {Blue,Green,Red,Yellow,BLACK};

struct SceneSegment
{
  int label;
  int seg_size;
  int x,y;
  cv::Rect bbox;
  float area;
  cv::Scalar color;
  float prob_bgfg;
  cv::Mat mask;
  cv::Mat bmask;
  float saliency;
  Stimulus stimulus;
  double CCT;
};

class SaliencyExtractor
{

    cuda::GpuMat gpuImage;
    Mat Img,LMSImg;
    Mat histogram,histogramIndex,spDist,neighbours,meanX,meanY;
    Mat SM,SalImg,GSal,LSal;
    vector<Mat> LMS;
    float _sigma;
    int colors[BINS*BINS*BINS][3];
    int histSize1,histSize2,histSize3;
    int _logSize,_logSize2;
    vector<Mat> channels;
    vector<Rect> fixations;
    vector<SceneSegment> segments;
    AttentionModel AM;
    cv::Mat msSegImg;
    double meanSal;
    cv::Scalar meanColor;
    float stimulus_weight[5] = {0.25,0.25,0.25,0.25,0};

    public:


    SaliencyExtractor(float sigma,AttentionModel am)
    {

        _sigma = sigma;
        histSize1 = BINS;
        histSize2 = BINS*BINS;
        histSize3 = BINS*histSize2;
        _logSize       = (int)log2(histSize1);
        _logSize2      = 2*_logSize;
        AM = am;

    }

    void compute(cv::Mat &img);
    void segmentScene();
    void computeHistogram();
    void computeSegmentToFixationCorrespondance();
    void computeSaliencyMap();
    void computeSaliencyMapAll();
    Mat computeGlobalSaliency();
    void computeSaliency();
    Mat inline getSaliencyMask() {return SM;}
    Mat inline getSaliencyMap() {return SalImg;}
    Mat inline getGlobalSaliencyMap() {return GSal;}
    Mat inline getLocalSaliencyMap() {return LSal;}
    vector<Mat> inline getOpponentSpace(){return LMS;}
    Mat inline getSegmentImage(){return msSegImg;}
    std::vector<SceneSegment> inline getSalientRegions() {return segments;}
    void calcSpatialDist();
    void computeAttentionRegions(int cx,int cy,int wsize);
    void mergeSegments();
    void RGB2Lab(cv::Mat &,vector<cv::Mat> &);
    void BGR2OCS(cv::Mat &,vector<cv::Mat> &);
    Mat ifftShift(Mat &in);
    void meshgrid(const cv::Range &xgv, const cv::Range &ygv,cv::Mat &X, cv::Mat &Y);
    Mat logGaborMask(int x,int y,float sigma,float theta);
    double rgb2cct(int R,int G,int B);
    void cct2rgb(double R,double * RGB);
};

#endif // SALIENTEXTRACTOR_H
