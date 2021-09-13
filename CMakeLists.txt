# Catkin User Guide: http://www.ros.org/doc/groovy/api/catkin/html/user_guide/user_guide.html
# Catkin CMake Standard: http://www.ros.org/doc/groovy/api/catkin/html/user_guide/standards.html
cmake_minimum_required(VERSION 2.8.3)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(CMAKE_VERBOSE_MAKEFILE ON)
project(semantic_mapping)

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/CMakeModules ${CMAKE_MODULE_PATH})

find_package(OpenCV 3 REQUIRED PATHS /usr/local/share/OpenCV NO_DEFAULT_PATH)
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories( /usr/local/lib )
#find_package(PCL 1.8 REQUIRED)
#include_directories(${PCL_INCLUDE_DIRS})
#link_directories(${PCL_LIBRARY_DIRS})
#add_definitions(${PCL_DEFINITIONS})

# Load catkin and all dependencies required for this package
find_package(catkin REQUIRED COMPONENTS
  pcl_ros
  rosconsole
  cmake_modules 
  laser_geometry 
  sensor_msgs 
  visualization_msgs 
  std_srvs
  pcl_conversions
  eigen_conversions
  octomap_ros
  octomap_msgs
)

catkin_package(
    DEPENDS eigen libpcl-all-dev
    CATKIN_DEPENDS pcl_ros laser_geometry sensor_msgs std_srvs message_runtime visualization_msgs
    INCLUDE_DIRS include
    LIBRARIES ${PROJECT_NAME} 
)


include_directories(include)
include_directories(${catkin_INCLUDE_DIRS})

find_package(OpenMP)
find_package(OpenGL)
find_package(GLUT)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")

include_directories(${Boost_INCLUDE_DIRS})
link_directories( ${catkin_LIB_DIR} )

include_directories("/home/sg/libs/dlib")
#include_directories(libs/egbs)
#file(GLOB EBIS_SRC libs/egbs/egbis/*.h libs/egbs/egbis/*.cpp)
#add_library(egbs  libs/egbs/egbis.cpp ${EBIS_SRC})
#target_link_libraries(egbs ${catkin_LIBRARIES})

#include_directories(libs/MRF2.2)
#file(GLOB MRF_SRC libs/MRF2.2/*.h libs/MRF2.2/*.cpp)
#add_library(mrf  libs/MRF2.2/mrf.cpp ${MRF_SRC})
#target_link_libraries(mrf ${catkin_LIBRARIES})

include_directories(libs/image_geometry/include)
add_library(img_geom  libs/image_geometry/src/pinhole_camera_model.cpp)
target_link_libraries(img_geom ${OpenCV_LIBRARIES})

include_directories(libs/cv_bridge/include)
add_library(cvbridge  libs/cv_bridge/src/cv_bridge.cpp)
target_link_libraries(cvbridge ${OpenCV_LIBRARIES})

add_library(semExtractor src/semanticextractor.cpp)
target_link_libraries(semExtractor ${Boost_LIBRARIES} ${OpenCV_LIBRARIES} ${catkin_LIBRARIES})
message("OPENCV_LIBS= ${OpenCV_LIBS} opencv_gpu")



add_library(semviz src/SemVizGlut.cpp)
target_link_libraries(semviz  ${Boost_LIBRARIES} ${catkin_LIBRARIES} ${GLUT_LIBRARIES} ${OPENGL_LIBRARIES})

add_library(salex src/SaliencyExtractor.cpp)
target_link_libraries(salex ${OpenCV_LIBRARIES} )

add_library(odet src/ObjectDetector.cpp)
target_link_libraries(odet ${OpenCV_LIBRARIES} )

include(/home/sg/libs/dlib/dlib/cmake)

add_executable(semMapper src/semMapper.cpp)
target_link_libraries(semMapper cvbridge img_geom semExtractor salex odet dlib semviz ${Boost_LIBRARIES} ${OpenCV_LIBRARIES}  ${catkin_LIBRARIES} ${GLUT_LIBRARIES} ${OPENGL_LIBRARIES} )

add_executable(evaluateSal src/evaluateSaliency.cpp)
target_link_libraries(evaluateSal salex dlib ${OpenCV_LIBRARIES} ${Boost_LIBRARIES} ${catkin_LIBRARIES})

add_executable(evaluateObj src/evaluateObjectDetection.cpp)
target_link_libraries(evaluateObj odet dlib ${OpenCV_LIBRARIES} ${Boost_LIBRARIES} ${catkin_LIBRARIES})

add_executable(semtest src/test.cpp)
target_link_libraries(semtest semviz ${OpenCV_LIBRARIES} ${Boost_LIBRARIES} ${catkin_LIBRARIES} ${GLUT_LIBRARIES} ${OPENGL_LIBRARIES} )
