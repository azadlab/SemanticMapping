/*
Copyright (C) 2015 Yasutomo Kawanishi
Copyright (C) 2013 Christoffer Holmstedt
Copyright (C) 2010 Salik Syed
Copyright (C) 2006 Pedro Felzenszwalb

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307 USA
*/
#include "egbis.h"
#include <opencv2/opencv.hpp>
#include "egbis/segment-image.h"
#include "egbis/misc.h"
#include "egbis/image.h"

/****
 * OpenCV C++ Wrapper using the Mat class
 ***/
image<lab>* convertRGBToNativeImage(const cv::Mat& input){
    int w = input.cols;
    int h = input.rows;
    image<lab> *im = new image<lab>(w,h);

    for(int i=0; i<h; i++)
    {
        for(int j=0; j<w; j++)
        {
            lab curr;
            cv::Vec3f intensity = input.at<cv::Vec3f>(i,j);
            curr.l = intensity.val[0];
            curr.a = intensity.val[1];
            curr.b = intensity.val[2];
            im->data[i*w+j] = curr;
        }
    }
    return im;
}

image<float>* convertDepthToNativeImage(const cv::Mat& input){
    int w = input.cols;
    int h = input.rows;
    image<float> *im = new image<float>(w,h);

    for(int i=0; i<h; i++)
    {
        for(int j=0; j<w; j++)
        {
            float depth = input.at<float>(i,j);
            im->data[i*w+j] = depth;
        }
    }
    return im;
}

cv::Scalar inline randomColor( cv::RNG& rng ) {
    int icolor = (unsigned) rng;
    return cv::Scalar( icolor&255, (icolor>>8)&255, (icolor>>16)&255 );
}


void convertNativeToMat(image<u_int16_t>* labels,cv::Mat &rgbImg,cv::Mat &lblImg){
    int width = labels->width();
    int height = labels->height();
    rgbImg = cv::Mat(cv::Size(width,height),CV_8UC3);
    lblImg = cv::Mat(cv::Size(width,height),CV_16UC1);

    cv::RNG ran_gen(12345);

    std::vector<cv::Scalar> colors;
    for (int i = 0; i < width*height; i++)
      colors.push_back(randomColor(ran_gen));

    for(int i =0; i<height; i++){
        for(int j=0; j<width; j++){

            u_int16_t lbl = labels->data[i*width+j];
            lblImg.at<u_int16_t>(i,j) = lbl;
            cv::Scalar color = colors[(int)lbl];
            rgbImg.at<cv::Vec3b>(i,j)[0] = color(0);
            rgbImg.at<cv::Vec3b>(i,j)[1] = color(1);
            rgbImg.at<cv::Vec3b>(i,j)[2] = color(2);
        }
    }

}

void  runEgbisOnMat(const cv::Mat rgbImg, cv::Mat depthImg,cv::Mat &segImg,cv::Mat &lblImg,float sigma, float k, int min_size, int *numccs) {
    int w = rgbImg.cols;
    int h = rgbImg.rows;

    // 1. Convert RGB to native format
    image<lab> *colorImage = convertRGBToNativeImage(rgbImg);
    // 2. Convert depth to native format
    image<float> *dImage = convertDepthToNativeImage(depthImg);
    // 3. Run egbis algoritm
    image<u_int16_t> *segmentedImage = segment_image(colorImage, dImage,sigma, k, min_size, numccs);

    convertNativeToMat(segmentedImage,segImg,lblImg);

    delete colorImage,dImage;
	delete segmentedImage;

}
