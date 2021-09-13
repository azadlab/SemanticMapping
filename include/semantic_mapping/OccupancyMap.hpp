

void OccupancyMap::addCloud(PCLPointCloud &cloud,Eigen::Matrix4d sensorToWorld)
{

    pcl::PassThrough<PCLPointType> pass;
    pass.setFilterFieldName("z");
    pass.setFilterLimits(1, 10);
    pass.setInputCloud(cloud.makeShared());
    pass.filter(cloud);

    pcl::transformPointCloud(cloud, cloud, sensorToWorld);
    point3d origin(sensorToWorld(0,3),sensorToWorld(1,3),sensorToWorld(2,3));
    insertScan(origin, cloud);

}


void OccupancyMap::insertScan(point3d &sensorOrigin, PCLPointCloud& cloud){


  if (!m_octree->coordToKeyChecked(sensorOrigin, m_updateBBXMin)
    || !m_octree->coordToKeyChecked(sensorOrigin, m_updateBBXMax))
  {
    ROS_ERROR_STREAM("ERROR WHILE Generating key "<<sensorOrigin);
  }


  KeySet free_cells, occupied_cells;

  std::vector<int> indices;
  int idx = 0;

  for (PCLPointCloud::const_iterator it = cloud.begin(); it != cloud.end(); ++it){
    point3d point(it->x, it->y, it->z);


    if ((m_maxRange < 0.0) || ((point - sensorOrigin).norm() <= m_maxRange) ) {


      if (m_octree->computeRayKeys(sensorOrigin, point, m_keyRay)){
        free_cells.insert(m_keyRay.begin(), m_keyRay.end());
//        m_octree->setNodeColor(m_keyRay,0,255,0);
      }

      OcTreeKey key;
      if (m_octree->coordToKeyChecked(point, key)){
        occupied_cells.insert(key);
        indices.push_back(++idx);
        updateMinKey(key, m_updateBBXMin);
        updateMaxKey(key, m_updateBBXMax);

        octomap::ColorOcTreeNode* node = m_octree->search(key);
        if(node==nullptr)
            m_octree->setNodeColor(key,it->r,it->g,it->b);
        else
        if(!node->isColorSet() || (base_color.r==node->getColor().r & base_color.g==node->getColor().g & base_color.b==node->getColor().b))
            m_octree->setNodeColor(key,it->r,it->g,it->b);
      }
    } else {
      point3d new_end = sensorOrigin + (point - sensorOrigin).normalized() * m_maxRange;
      if (m_octree->computeRayKeys(sensorOrigin, new_end, m_keyRay)){
        free_cells.insert(m_keyRay.begin(), m_keyRay.end());
        ROS_ERROR_STREAM("RAY LONGER THAN ENDPOINT -- EXTENDING MAP \n"<<new_end);
        ROS_INFO("COLOR (R,G,B):(%d,%d,%d)\n",it->r,it->g,it->b);
        octomap::OcTreeKey endKey;
        if (m_octree->coordToKeyChecked(new_end, endKey)){
          updateMinKey(endKey, m_updateBBXMin);
          updateMaxKey(endKey, m_updateBBXMax);
        } else{
          ROS_ERROR_STREAM("Could not generate Key for endpoint "<<new_end);
        }


      }
    }
  }


  for(KeySet::iterator it = free_cells.begin(), end=free_cells.end(); it!= end; ++it){
    if (occupied_cells.find(*it) == occupied_cells.end()){
      m_octree->updateNode(*it, false);
    }
  }

  idx = 0;
  for (KeySet::iterator it = occupied_cells.begin(), end=free_cells.end(); it!= end; it++) {
      m_octree->updateNode(*it, true);
//      octomap::ColorOcTreeNode* node = m_octree->search(*it);
//      if(!node->isColorSet() || (base_color.r==node->getColor().r & base_color.g==node->getColor().g & base_color.b==node->getColor().b))
//      {
//          m_octree->setNodeColor(*it,nonground.points[idx].r,nonground.points[idx].g,nonground.points[idx].b);

//          idx++;}
      }


}
