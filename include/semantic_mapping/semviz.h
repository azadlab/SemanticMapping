/*

Class Name: SemViz
Author Name: J. Rafid S.
Author URI: www.azaditech.com
Description: Supporting class for visualizing semantic Map.
Relevant Publication: Siddiqui, J. R. , Andreasson, H. , Driankov, D. & Lilienthal, A. J. (2016). Towards visual mapping in industrial environments: a heterogeneous task-specific and saliency driven approach. IEEE International Conference on Robotics and Automation (ICRA). , Stockholm, Sweden, May, 2016.

*/

#ifndef SEMVIZ_HH
#define SEMVIZ_HH
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/StdVector>
#include <semantic_mapping/SemVizGlut.hh>
#include <pthread.h>
#include <semantic_mapping/semmap.h>
#include <octomap/octomap.h>

class SemViz {

    public:
    SemVizGlut *win3D;
    SemVizGlutPointCloudColor gl_points;
    SemVizGlutPointCloudColor gl_particles;
    SemVizGlutPointCloudColor gl_pointcloud;
    SemVizGlutSetOfLines gl_laserlines;
    SemVizGlutCuboids gl_cuboids;
    SemVizGlutSpheres gl_spheres;
    SemVizGlutEllipsoids gl_ellipsoids;

    SemViz(bool allocate_new_window=true)

    {
        if(allocate_new_window)
        {
          win3D = new SemVizGlut();
          int argc=0;
          char** argv = NULL;
          win3D->win_run(&argc,argv);
        }
        else
        {
        win3D = NULL;
        }

        gl_points.setPointSize(5);
        gl_particles.setPointSize(10);

    }
    virtual ~SemViz () {
        if(win3D!=NULL) {
        win3D->win_close();
        delete win3D;
        }
    }

    void startEventLoop() {

    }

    void repaint(){
        win3D->repaint();
    }

    void clear(){
      win3D->clearScene();
    }

    void clearTrajectoryPoints(){
      gl_points.clear();

    }

    void addTrajectoryPoint(float x, float y, float z, float R=1.0, float G = 1.0, float B = 1.0){
      gl_points.push_back(x, y, z, R ,G,B);

    }
    void displayTrajectory(){
      win3D->addObject(&gl_points);

    }

    void clearParticles(){ gl_particles.clear();}
    void addParticle(float x, float y, float z, float R=1.0, float G = 1.0, float B = 1.0){
        gl_particles.push_back(x, y, z, R ,G,B);
    }
    void displayParticles(){
      win3D->addObject(&gl_particles);
    }
    void setCameraPointingToPoint(double x, double y, double z) {
      win3D->setCameraPointingToPoint(x,y,z);
    }
    void setCameraPointing(double x, double y, double z) {
      win3D->setCameraPointingToPoint(x,y,z);
    }



    /**
      * Add the laser scan to the scen
      */
    void addScan(Eigen::Vector3d orig, pcl::PointCloud<pcl::PointXYZ> &cloud, double R=1.0,double G=1.0,double B=1.0){

      gl_laserlines.clear();
      for(unsigned int i=0;i<cloud.points.size();i+=2){
        gl_laserlines.appendLine(orig(0),orig(1),orig(2), cloud.points[i].x, cloud.points[i].y, cloud.points[i].z);
      }
      gl_laserlines.setColor(R,G,B);
      win3D->addObject(&gl_laserlines);
    }
    void addPointCloud(pcl::PointCloud<pcl::PointXYZ> &cloud, double R=1.0,double G=1.0,double B=1.0){

      gl_pointcloud.setPointSize(3);
      for(unsigned int i=0;i<cloud.points.size();i+=2){
        gl_pointcloud.push_back(cloud.points[i].x, cloud.points[i].y, cloud.points[i].z,R,G,B);
      }

      win3D->addObject(&gl_pointcloud);
    }


    void addPointCloud(pcl::PointCloud<pcl::PointXYZRGB> &cloud){
        if(win3D == NULL) return;
gl_pointcloud.clear();
      gl_pointcloud.setPointSize(3);


      for(unsigned int i=0;i<cloud.points.size();i++){

        if(cloud.points[i].x==cloud.points[i].x && cloud.points[i].y==cloud.points[i].y && cloud.points[i].z==cloud.points[i].z)
        gl_pointcloud.push_back(cloud.points[i].x, cloud.points[i].y, cloud.points[i].z,cloud.points[i].r/255.0,cloud.points[i].g/255.0,cloud.points[i].b/255.0);
      }

      win3D->addObject(&gl_pointcloud);

    }

    void viewSemMap(SemMapPtr &sm){

        if(win3D == NULL) return;
        win3D->clearScene();
        gl_cuboids.clear();

        octomap::ColorOcTree* omap = sm->occmap.getOccMap();

        int numRegions = sm->semRegions.size();

       for(octomap::ColorOcTree::iterator it = omap->begin(sm->occmap.getTreeDepth()),end=omap->end();it!=end;++it)
        {

            if (omap->isNodeOccupied(*it)){
              double z = it.getZ();
              unsigned idx = it.getDepth();
              double size = omap->getNodeSize(idx);
              double x = it.getX();
              double y = it.getY();

                if ((it.getDepth() == omap->getTreeDepth() +1) && sm->occmap.isSpeckleNode(it.getKey())){
                  ROS_DEBUG("Ignoring single speckle at (%f,%f,%f)", x, y, z);
                  continue;
                }
            double minDist=INFINITY;

            octomap::ColorOcTreeNode node = *it;
            octomap::ColorOcTreeNode::Color C = node.getColor();
            Eigen::Vector3d color(C.r/255.0,C.g/255.0,C.b/255.0);

            /*Eigen::Vector3d color;

            for(int i=0;i<numRegions;i++)
            {
                SemRegion R = sm->semRegions[i];

                vector<Eigen::Matrix4d> poses = R.getMapOccurances();
                for(int j=0;j<poses.size();j++)
                {
                    Eigen::Matrix4d pose = poses[j];

                    double dist = sqrt( (pose(0,3)-x)*(pose(0,3)-x) + (pose(1,3)-y)*(pose(1,3)-y)  );
                    if(minDist>dist)
                    {
                        minDist = dist;
                        color(0) = sm->colors[R.getLabel()][0];
                        color(1) = sm->colors[R.getLabel()][1];
                        color(2) = sm->colors[R.getLabel()][2];
                    }

                }

            }

            if(minDist>0.4)
                color = Eigen::Vector3d(1,0,0);*/
            SemVizGlutCuboid obj(1,1,true,10);
            obj.setLocation(x,y,z);
            obj.setScale(size);
            obj.setCov(Eigen::Matrix3d::Identity());
            obj.setColor(color[0],color[1],color[2],0.7);
            obj.enableDrawSolid3D(true);
            gl_cuboids.push_back(obj);

        }
       }

        win3D->addObject(&gl_cuboids);

        //win3D->addObject(&gl_ellipsoids);
        win3D->repaint();

    }

    void viewSemMapRegions(SemMapPtr &sm){

           if(win3D == NULL) return;
           win3D->clearScene();
           gl_cuboids.clear();


           int numRegions = sm->semRegions.size();


//cout<<"Plotting Map with Regions="<<numRegions<<endl;
           for(int i=0;i<numRegions;i++)
           {
               SemRegion R = sm->semRegions[i];
//cout<<"Plotting Region with Label="<<R.getLabel()<<endl;

               vector<cv::Mat> poses = R.getMapOccurances();
               for(int j=0;j<poses.size();j++)
               {
                   cv::Mat pose = poses[j];
                   SemVizGlutCuboid obj(1,1,true,10);
                   obj.setLocation(pose.at<float>(0),pose.at<float>(1),pose.at<float>(2));
                   obj.setScale(0.2);
                   obj.setCov(Eigen::Matrix3d::Identity());
                   obj.setColor(sm->colors[R.getSemLabel()][0],sm->colors[R.getSemLabel()][1],sm->colors[R.getSemLabel()][2],0.7);
                   obj.enableDrawSolid3D(true);
                   gl_cuboids.push_back(obj);

               }

           }
           win3D->addObject(&gl_cuboids);
           win3D->repaint();

       }

};

#endif
