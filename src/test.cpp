/*

Author Name: J. Rafid S.
Author URI: www.azaditech.com
Relevant Publication: Siddiqui, J. R. , Andreasson, H. , Driankov, D. & Lilienthal, A. J. (2016). Towards visual mapping in industrial environments: a heterogeneous task-specific and saliency driven approach. IEEE International Conference on Robotics and Automation (ICRA). , Stockholm, Sweden, May, 2016.

*/

#include <semantic_mapping/semviz.h>

int main(int argc, char **argv){

    string mappath(argv[1]);
    string mapname(argv[2]);

    SemMapPtr semmap = boost::make_shared<SemMap>(0.2,100);
    if(!semmap->readMap(mappath,mapname))
    {
        cout<<"Problem reading map"<<endl;
        return -1;
    }
cout<<"map read successfully"<<endl;
cout<<"map size="<<semmap->semRegions.size()<<endl;
    SemViz * viewer;
    viewer = new SemViz(true);
    viewer->win3D->start_main_loop_own_thread();
    viewer->viewSemMap(semmap);
    cv::Mat init_pos = semmap->getInitPosition();
    float x = init_pos.at<float>(0);
    float y = init_pos.at<float>(1);
    float z = init_pos.at<float>(2);
    viewer->win3D->setOrigin(x,y,z);
    viewer->win3D->draw_origin();
    viewer->setCameraPointing(x,y,z+3);
    viewer->repaint();
     while(viewer->win3D->isOpen()){
         //viewer->viewSemMap(semmap);
         usleep(1000);
     }
delete viewer;
    return 0;
}

