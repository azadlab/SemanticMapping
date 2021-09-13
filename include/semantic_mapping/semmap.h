/*

Class Name: SemMap
Author Name: J. Rafid S.
Author URI: www.azaditech.com
Relevant Publication: Siddiqui, J. R. , Andreasson, H. , Driankov, D. & Lilienthal, A. J. (2016). Towards visual mapping in industrial environments: a heterogeneous task-specific and saliency driven approach. IEEE International Conference on Robotics and Automation (ICRA). , Stockholm, Sweden, May, 2016.

*/

#include<semantic_mapping/semregion.h>
#include <boost/ptr_container/ptr_container.hpp>
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <semantic_mapping/OccupancyMap.h>
#include <semantic_mapping/utils.h>
#include <fstream>
#include <boost/serialization/serialization.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>


#ifndef CV_SERIAL
#define CV_SERIAL
#include<semantic_mapping/cv_serializer.h>
#endif

using namespace std;

#ifndef SEMMAP_H
#define SEMMAP_H

typedef boost::ptr_vector<SemRegion> Regions;

class SemMap
{

    float resolution;
    float max_range;
    cv::Mat init_pos;

public:

    Regions semRegions;
    OccupancyMap occmap;
    Eigen::MatrixXd Dict;
    vector<cv::Vec3f> colors;
    SemMap(float res,float mxr):occmap(res,mxr)
    {
        resolution = res;
        max_range= mxr;
        getColors(colors,150);
        init_pos = cv::Mat::zeros(1,3,CV_32F);

    }
    void setInitPosition(float x,float y,float z)
    {
        init_pos.at<float>(0) = x;
        init_pos.at<float>(1) = y;
        init_pos.at<float>(2) = z;
    }
    cv::Mat getInitPosition()
    {
        return init_pos;
    }

    bool saveMap(string path,string mapname)
    {
        ofstream ofs;
        ofs.open(path+mapname+".sem",ios::binary);
        try{
        boost::archive::text_oarchive oa(ofs);
        size_t numRegions = semRegions.size();
        oa << numRegions;
        oa << resolution;
        oa << max_range;
        oa << init_pos;
        cout<<"Saving "<<numRegions<<" Semantic Regions to map file"<<endl;
        for(size_t i=0;i<semRegions.size();i++)
            oa << semRegions[i];

        occmap.saveMap(path+mapname+".ocm");
        ofs.close();
        }catch(exception ex){cout<<"ERROR:COULD NOT WRITE SEMANTIC REGIONS";return false;}
        return true;
    }


    bool readMap(string path,string mapname)
    {
        ifstream ifs;
        ifs.open(path+mapname+".sem",ios::binary);
        try{
            if(!ifs)
            {
                cout<<"Can't read map file"<<endl;
                return false;
            }
            boost::archive::text_iarchive ia(ifs);
            size_t numRegions;
            ia >> numRegions;
            ia >> resolution;
            ia >> max_range;
            ia >> init_pos;
            semRegions.clear();
            SemRegion S;
            for(size_t i=0;i<numRegions;i++)
            {
                ia >> S;
                semRegions.push_back(new SemRegion(S));
            }
            occmap.readMap(path+mapname+".ocm");

            ifs.close();
        }catch(exception const &ex){cout<<"Exception:"<<ex.what()<<endl;return false;}
        return true;
    }

    ~SemMap()
    {

    }

};

typedef boost::shared_ptr<SemMap> SemMapPtr;
#endif // SEMMAP_H
