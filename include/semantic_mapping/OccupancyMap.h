/*

Class Name: OccupancyMap
Author Name: J. Rafid S.
Author URI: www.azaditech.com
Description: Class for creating OccupancyMap.
Relevant Publication: Siddiqui, J. R. , Andreasson, H. , Driankov, D. & Lilienthal, A. J. (2016). Towards visual mapping in industrial environments: a heterogeneous task-specific and saliency driven approach. IEEE International Conference on Robotics and Automation (ICRA). , Stockholm, Sweden, May, 2016.

*/

#ifndef OCCUPANCYMAP_H
#define OCCUPANCYMAP_H
#include<ros/ros.h>
#include<octomap/octomap.h>
#include<pcl/point_types.h>
#include <pcl/point_cloud.h>
#include<pcl/filters/passthrough.h>
#include <pcl/filters/extract_indices.h>
#include <pcl_ros/transforms.h>
#include <nav_msgs/OccupancyGrid.h>
#include <octomap/OcTreeKey.h>
#include <tf/transform_listener.h>
#include <tf/message_filter.h>
#include <message_filters/subscriber.h>
#include <octomap_msgs/Octomap.h>
#include <octomap_msgs/GetOctomap.h>
#include <octomap_msgs/BoundingBoxQuery.h>
#include <octomap_msgs/conversions.h>
#include <octomap_ros/conversions.h>
#include <octomap/ColorOcTree.h>


typedef pcl::PointXYZRGB PCLPointType;
typedef pcl::PointCloud<PCLPointType> PCLPointCloud;

using namespace octomap;

class OccupancyMap{

ColorOcTree *m_octree;
//OcTree *m_octree;
OcTreeKey m_updateBBXMin;
OcTreeKey m_updateBBXMax;
octomap::KeyRay m_keyRay;
double m_maxRange;
double m_treeDepth;
octomap::ColorOcTreeNode::Color base_color;
public:

OccupancyMap(float resolution,double max_range):base_color(255,0,0)
{
    m_octree = new ColorOcTree(resolution);
    m_maxRange = max_range;
    m_treeDepth = m_octree->getTreeDepth();

}
~OccupancyMap()
{
    delete m_octree;
}
    virtual void addCloud(PCLPointCloud&,Eigen::Matrix4d);
    virtual void insertScan(point3d&,PCLPointCloud&);
    inline ColorOcTree* getOccMap(){return m_octree;}
    inline void setOccMap(ColorOcTree* coct){m_octree=coct;}
    inline int getMapSize(){return m_octree->size();}
    inline bool saveMap(string filename){return m_octree->write(filename);}
    inline bool readMap(string filename){
        AbstractOcTree* rt = AbstractOcTree::read(filename);
        if(rt)
        {
            m_octree = dynamic_cast<ColorOcTree*>(rt);
            return true;
        }
        return false;
    }
    double inline getTreeDepth(){return m_treeDepth;}
    void inline getMapMin(double& x,double &y,double &z){m_octree->getMetricMin(x,y,z);}
    void inline getMapMax(double& x,double &y,double &z){m_octree->getMetricMax(x,y,z);}
    void inline getMapSize(double& x,double &y,double &z){m_octree->getMetricSize(x,y,z);}
    inline static void updateMinKey(const octomap::OcTreeKey& in, octomap::OcTreeKey& min){
      for (unsigned i=0; i<3; ++i)
        min[i] = std::min(in[i], min[i]);
    };
    bool isSpeckleNode(const OcTreeKey&nKey) const {
      OcTreeKey key;
      bool neighborFound = false;
      for (key[2] = nKey[2] - 1; !neighborFound && key[2] <= nKey[2] + 1; ++key[2]){
        for (key[1] = nKey[1] - 1; !neighborFound && key[1] <= nKey[1] + 1; ++key[1]){
          for (key[0] = nKey[0] - 1; !neighborFound && key[0] <= nKey[0] + 1; ++key[0]){
            if (key != nKey){
              OcTreeNode* node = m_octree->search(key);
              if (node && m_octree->isNodeOccupied(node)){
                neighborFound = true;
              }
            }
          }
        }
      }
      return neighborFound;
    }

    inline static void updateMaxKey(const octomap::OcTreeKey& in, octomap::OcTreeKey& max){
      for (unsigned i=0; i<3; ++i)
        max[i] = std::max(in[i], max[i]);
    };
    void getSemRegionForPoint(point3d,SemRegion&);
};
#include <semantic_mapping/OccupancyMap.hpp>
#endif // OCCUPANCYMAP_H
