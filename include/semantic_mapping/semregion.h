/*

Class Name: SemRegion
Author Name: J. Rafid S.
Author URI: www.azaditech.com
Description: Semantic Region Extractor
Relevant Publication: Siddiqui, J. R. , Andreasson, H. , Driankov, D. & Lilienthal, A. J. (2016). Towards visual mapping in industrial environments: a heterogeneous task-specific and saliency driven approach. IEEE International Conference on Robotics and Automation (ICRA). , Stockholm, Sweden, May, 2016.

*/

#ifndef SEMREGION_H
#define SEMREGION_H

#include <iostream>
#include <string>
#include <opencv2/core.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/vector.hpp>

const int BINS = 8;

using namespace std;

enum SemLabel  {Pallet,Pillar,Depository,Unknown};

class Appearance
{

public:
    cv::Mat meanColor;
    cv::Mat Orientation;


    Appearance()
    {

        Orientation = cv::Mat::zeros(1,31,CV_32F);
        meanColor = cv::Mat::zeros(1,3,CV_8U);


    }
    friend class boost::serialization::access;
    template<class Archive>
    void serialize(Archive& ar, const unsigned int version)
    {
      ar & BOOST_SERIALIZATION_NVP(meanColor);
      ar & BOOST_SERIALIZATION_NVP(Orientation);
    }
};

class SemRegion:boost::noncopyable
{
    SemLabel semlabel;
    vector<cv::Mat> mapOccurances;
    vector<Appearance> appearances;


    friend class boost::serialization::access;
    template <class Archive>
    void serialize(Archive& ar, const unsigned int version)
    {
      ar & BOOST_SERIALIZATION_NVP(semlabel);
      ar & BOOST_SERIALIZATION_NVP(mapOccurances);
      ar & BOOST_SERIALIZATION_NVP(appearances);

    }
public:

    bool isvalid=true;
    SemRegion(){}

    SemRegion(const SemRegion &v)
    {
        this->semlabel = v.semlabel;
        this->mapOccurances = v.mapOccurances;
        this->appearances = v.appearances;
        isvalid = v.isvalid;

    }

    vector<cv::Mat> inline getMapOccurances(void) {return this->mapOccurances;}
    void inline addPose(cv::Mat pose){this->mapOccurances.push_back(pose);}
    void inline addPoses(std::vector<cv::Mat> poses){this->mapOccurances.insert(mapOccurances.end(),poses.begin(),poses.end());}
    void inline addAppearance(Appearance A) { this->appearances.push_back(A);}
    void inline addAppearances(std::vector<Appearance> AA) { this->appearances.insert(appearances.end(),AA.begin(),AA.end());}
    vector<Appearance> inline getAppearances() { return this->appearances;}
    void inline setSemLabel(SemLabel sem) {semlabel=sem;}
    SemLabel inline getSemLabel(void) {return semlabel;}

};

#endif // SEMREGION_H
