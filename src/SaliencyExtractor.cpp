/*

Author Name: J. Rafid S.
Author URI: www.azaditech.com
Relevant Publication: Siddiqui, J. R. , Andreasson, H. , Driankov, D. & Lilienthal, A. J. (2016). Towards visual mapping in industrial environments: a heterogeneous task-specific and saliency driven approach. IEEE International Conference on Robotics and Automation (ICRA). , Stockholm, Sweden, May, 2016.

*/


#include <semantic_mapping/SaliencyExtractor.h>

void SaliencyExtractor::compute(cv::Mat &img)
{
    this->Img = img;

    if(AM == AttentionModel::RAND_FIX)
        computeAttentionRegions(0,floor(Img.rows/2),floor(Img.cols/2));
    else if(AM == AttentionModel::SCAN_FIX)
        computeAttentionRegions(0,0,floor(Img.cols/3));

    segmentScene();

    if(AM == AttentionModel::RAND_FIX)
        computeSaliencyMap();
    else if(AM == AttentionModel::SCAN_FIX)
        computeSaliencyMapAll();


}

void SaliencyExtractor::computeHistogram(){



    Mat  C1shift, C2shift, C3shift;

    double minC1, maxC1, minC2, maxC2, minC3, maxC3;


    minMaxLoc(channels[0], &minC1, &maxC1);
    minMaxLoc(channels[1], &minC2, &maxC2);
    minMaxLoc(channels[2], &minC3, &maxC3);

    float tempC1 = (255 - maxC1 + minC1) / (maxC1 - minC1 + 1e-3);
    float tempC2 = (255 - maxC2 + minC2) / (maxC2 - minC2 + 1e-3);
    float tempC3 = (255 - maxC3 + minC3) / (maxC3 - minC3 + 1e-3);

    C1shift          = Mat::zeros(1, 256, CV_32SC1);
    C2shift          = Mat::zeros(1, 256, CV_32SC1);
    C3shift          = Mat::zeros(1, 256, CV_32SC1);

    for (int i = 0; i < 256; i++) {

        C1shift.at<int>(0,i) = tempC1 * (i - minC1) - minC1;
        C2shift.at<int>(0,i) = tempC2 * (i - minC2) - minC2;
        C3shift.at<int>(0,i) = tempC3 * (i - minC3) - minC3;

    }


    tempC1   = float(maxC1 - minC1)/histSize1;
    tempC2   = float(maxC2 - minC2)/histSize1;
    tempC3   = float(maxC3 - minC3)/histSize1;

    float sC1 = float(maxC1 - minC1)/histSize1/2 + minC1;
    float sC2 = float(maxC2 - minC2)/histSize1/2 + minC2;
    float sC3 = float(maxC3 - minC3)/histSize1/2 + minC3;

    for (int i = 0; i < histSize3; i++) {

        int C1pos = i % histSize1;
        int C2pos = i % histSize2 / histSize1;
        int C3pos = i / histSize2;

        colors[i][0] = (C1pos * tempC1 + sC1);
        colors[i][1] = (C2pos * tempC2 + sC2);
        colors[i][2] = (C3pos * tempC3 + sC3);

    }


    histogramIndex          = Mat::zeros(Img.rows, Img.cols, CV_32SC1);
    histogram               = Mat::zeros(1, histSize3, CV_32SC1);
    //meanX               = Mat::zeros(1, histSize3, CV_32SC1);
    //meanY               = Mat::zeros(1, histSize3, CV_32SC1);

    int*    histogramPtr    = histogram.ptr<int>(0);
    //int*    meanXPtr    = meanX.ptr<int>(0);
    //int*    meanYPtr    = meanY.ptr<int>(0);

    int*    C1shiftPtr       = C1shift.ptr<int>(0);
    int*    C2shiftPtr       = C2shift.ptr<int>(0);
    int*    C3shiftPtr       = C3shift.ptr<int>(0);

    int histShift = 8 - _logSize;

    for (int y = 0; y < Img.rows; y++) {

        int*    histogramIndexPtr   = histogramIndex.ptr<int>(y);

        uchar*    C1Ptr   = channels[0].ptr<uchar>(y);
        uchar*    C2Ptr   = channels[1].ptr<uchar>(y);
        uchar*    C3Ptr   = channels[2].ptr<uchar>(y);

        for (int x = 0; x < Img.cols; x++) {

            int C1pos                = (C1Ptr[x] + C1shiftPtr[C1Ptr[x]]) >> histShift;

            int C2pos                = (C2Ptr[x] + C2shiftPtr[C2Ptr[x]]) >> histShift;
            int C3pos                = (C3Ptr[x] + C3shiftPtr[C3Ptr[x]]) >> histShift;

            int index               = C1pos + (C2pos << _logSize) + (C3pos << _logSize2);

            histogramIndexPtr[x]    = index;

            histogramPtr[index]++;

            //meanXPtr[index]+=x;
            //meanYPtr[index]+=y;

        }
    }

}


void SaliencyExtractor::segmentScene() {


    Mat LMSImgUChar,grayImg;
    LMS.clear();
    BGR2OCS(Img,LMS);
    //LMS.push_back(Mat::zeros(Img.size(),CV_32FC1));
    merge(LMS,LMSImg);
    normalize(LMSImg,LMSImgUChar,0,1,NORM_MINMAX);

    LMSImgUChar.convertTo(LMSImgUChar,CV_8UC4,255);
    gpuImage.upload(LMSImgUChar);

    TermCriteria iterations = TermCriteria(CV_TERMCRIT_ITER, 2, 0);
    //cuda::cvtColor(gpuImageBGR,gpuImageLMS,CV_BGR2LMS,4);

    cuda::meanShiftSegmentation(gpuImage, msSegImg, 5, 10, 300, iterations);

    split(msSegImg, channels);
//imshow("MS_SEG",msSegImg);
//Mat gradImg,gx,gy;
//cvtColor(Img,grayImg,CV_BGR2GRAY);
//cv::Sobel(grayImg,gx,CV_16S,1,0,3);
//cv::Sobel(grayImg,gy,CV_16S,0,1,3);
//convertScaleAbs( gx, gx );
//convertScaleAbs( gy, gy );
//gx.convertTo(gx,CV_32FC1);
//gy.convertTo(gy,CV_32FC1);
//addWeighted(gx,0.5,gy,0.5,0,gradImg);
//float Grad = sum(gradImg)[0];
meanColor = cv::mean(msSegImg);

//convertToOpponentSpace();

    computeHistogram();

    //calcDist2();


    segments.clear();
    float img_size = Img.rows*Img.cols;

int k = 0;

    for (int i=0; i<histSize3; i++) {


            Mat boolImage = (histogramIndex == i);


            vector<vector<Point>> labelContours;
            findContours(boolImage, labelContours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
            long reg_size = histogram.at<int>(i);

            for (int idx = 0; idx < labelContours.size(); idx++) {
                SceneSegment s;


                if (reg_size > SEG_MIN) {

                    Rect bounds = boundingRect(labelContours[idx]);
                    float area = contourArea(labelContours[idx]);
                    s.mask = Mat::zeros(Img.size(), CV_8UC1);
                    drawContours(s.mask, labelContours, idx, Scalar(255), CV_FILLED);
                    s.mask.clone().convertTo(s.bmask,CV_32FC1,1/255.0);
                    long seg_size = sum(s.bmask)[0];
                    s.seg_size = seg_size;
                    s.label=i;
                    s.bbox = bounds;
                    s.area = area;
                    Scalar pos = mean(labelContours[idx]);
                    s.x = pos[0];
                    s.y = pos[1];

//                    Mat Gr = gradImg.mul(s.bmask);
//                    s.Gratio = sum(Gr)[0]/Grad;

                    float dist = sqrt((s.x-Img.cols/2)*(s.x-Img.cols/2) + (s.y-Img.rows/2)*(s.y-Img.rows/2));

                    s.prob_bgfg = exp(-dist/(_sigma*_sigma));

                    int B=colors[i][0],G=colors[i][1],R=colors[i][2];

                    s.color = cv::Scalar(B,G,R);
                    double cct = rgb2cct(R,G,B);
                    s.CCT = exp(-(cct*cct)/(5000*5000));

                    float Pb = B/255.0;
                    float Pg = G/255.0;
                    float Pr = R/255.0;

                    if(R>B & R>G)
                    {

                        stimulus_weight[0] *= 0.9;//Update Blue weight
                        stimulus_weight[1] *= 0.9; //Update Green weight
                        stimulus_weight[3] *= 0.9; //Update Yellow weight
//                        stimulus_weight[2] *= Pr;
                        s.stimulus = Red;
                    }
                    else if(R>B & G>B)
                    {
                        stimulus_weight[0] *= 0.9; //Update Blue weight
                        stimulus_weight[1] *= 0.9; //Update Green weight
//                        stimulus_weight[2] *= (Pr+Pg)/2;
//                        stimulus_weight[3] *= 1-(Pr+Pg)/2;
                        s.stimulus = Yellow;
                    }
                    else if(B>R & B>G)
                    {
//                        stimulus_weight[0] *= Pb;
                        stimulus_weight[1] *= 0.9; //Update Green weight
//                        stimulus_weight[2] *= 1-B;
//                        stimulus_weight[3] *= 1-B;

                        s.stimulus = Blue;
                    }
                    else if(G>B & G>R)
                    {
                        stimulus_weight[1] *= 0.9;
                        s.stimulus = Green;
                    }
                    else
                        s.stimulus=BLACK;

                    segments.push_back(s);
                }


            }

    }

    float Psum = stimulus_weight[0]+stimulus_weight[1]+stimulus_weight[2]+stimulus_weight[3];

    stimulus_weight[0] = stimulus_weight[0]/Psum;
    stimulus_weight[1] = stimulus_weight[1]/Psum;
    stimulus_weight[2] = stimulus_weight[2]/Psum;
    stimulus_weight[3] = stimulus_weight[3]/Psum;
    calcSpatialDist();

    gpuImage.release();

}


void SaliencyExtractor::computeSaliency()
{

GSal = computeGlobalSaliency();

normalize(GSal,GSal,0,1,NORM_MINMAX);
//    imshow("GSal",GSal);
//cout<<"weights: "<<stimulus_weight[0]<<", "<<stimulus_weight[1]<<", "<<stimulus_weight[2]<<", "<<stimulus_weight[3]<<", "<<endl;
    SalImg = Mat::zeros(Img.size(),CV_32FC1);
    LSal = Mat::zeros(Img.size(),CV_32FC1);
    float tot_sal=0;
    int nsegs = 0;
    for(int i=0;i<segments.size();i++)
    {
        SceneSegment& S = segments[i];

        if(S.seg_size > SEG_MIN)
        {

            double csal = 0;
            int ncount = 0;
            for(int j=0;j<segments.size();j++)
            {
                if(i==j)
                    continue;

                int nIdx = neighbours.at<int>(j);
                float sdist = spDist.at<float>(nIdx);
                SceneSegment S2 = segments[j];
                float edist = exp(-sdist/(_sigma*_sigma));
                if(S.label==S2.label)
                {


                    S.prob_bgfg *= edist;

                }
                else if(sdist > 1 )
                {
                    double cdist = 0;
                    SceneSegment NS = segments[nIdx];

                    Scalar diff;
                    diff = S.color-NS.color;
                    cdist = sqrt(sum(diff.mul(diff))[0]);

//                    if(S.stimulus==NS.stimulus)
//                        cdist = (S.CCT)*edist*NS.seg_size;
//                    else
                        cdist =(S.CCT)*cdist*edist;

//cout<<"(B,G,R)="<<S.color[0]<<", "<<S.color[1]<<", "<<S.color[2]<<endl;
//cout<<"S.CCT: "<<S.CCT<<", NS.CCT:"<<NS.CCT<<", cdist="<<cdist<<", S_segsize: "<<S.seg_size<<", NS_segsize:"<<NS.seg_size<<endl;


//                    if(S.stimulus == NS.stimulus)
//                        cdist = stimulus_weight[S.stimulus]*NS.seg_size*cdist;
//                    else
//                            cdist = (stimulus_weight[S.stimulus]*NS.seg_size-stimulus_weight[NS.stimulus]*S.seg_size)*cdist;
//                    cdist = cdist<0?0:cdist;

                    csal += cdist;


                    ncount++;
                }

            }

            if(ncount)
            {

                csal /= ncount;

                cv::Mat csmap;
                csmap = csal*S.bmask;

                LSal += csmap;

            }

        }
    }

LSal.setTo(0,LSal<0);
normalize(LSal,LSal,0,1,NORM_MINMAX);
//imshow("LSal",LSal);
SalImg = (LSal.mul(GSal));
    normalize(SalImg,SalImg,0,1,NORM_MINMAX);
//    cv::Mat bmask;
    for(int i=0;i<segments.size();i++)
    {
        SceneSegment& S = segments[i];

        if(S.seg_size > SEG_MIN)
        {

            S.saliency = sum(SalImg.mul(S.bmask))[0]/S.seg_size;
            tot_sal +=S.saliency;
            nsegs++;
        }
    }


    meanSal = tot_sal/nsegs;

//imshow("SalImg",SalImg);

}

void SaliencyExtractor::calcSpatialDist()
{
    int numSegs = segments.size();
    spDist  = Mat::zeros(numSegs, numSegs, CV_32FC1);

    for (int i = 0; i < numSegs; i++) {
        SceneSegment S = segments[i];
        spDist.at<float>(i,i)   = 0;
        int xi = S.x;
        int yi = S.y;

        for (int k = i + 1; k < numSegs; k++) {

            SceneSegment Sk = segments[k];

            int xk = Sk.x;
            int yk = Sk.y;
            float diff   = sqrt (pow(xi-xk,2) + pow(yi-yk,2) );


            spDist.at<float>(i,k)            = diff;

            spDist.at<float>(k,i)            = diff;



        }
    }

        cv::sortIdx(spDist,neighbours,CV_SORT_EVERY_ROW + CV_SORT_ASCENDING);
}

void SaliencyExtractor::computeAttentionRegions(int cx,int cy,int wsize)
{


    for(int i=cx;i<Img.cols-wsize+1;i+=10)
        for(int j=cy;j<Img.rows-wsize+1;j+=10)
        {
            Rect r(i,j,wsize,wsize);
            fixations.push_back(r);
        }

}

void SaliencyExtractor::computeSegmentToFixationCorrespondance()
{

    int ridx = rand()%fixations.size();
    Rect focus = fixations[ridx];

    vector<SceneSegment> focusedSegments;
    for(int i=0;i<segments.size();i++)
    {
        SceneSegment S = segments[i];
        Rect overlap = focus & S.bbox;
        int intsec_area = overlap.area();

        if( (intsec_area>0.7*focus.area() | intsec_area>0.9*S.area) & S.saliency>meanSal & S.area < 0.3*(Img.rows*Img.cols) & S.area>MIN_AREA)
            focusedSegments.push_back(S);
    }

    segments = focusedSegments;//mergeSegments(focusedSegments);

    //rectangle(Img,focus,cv::Scalar(0,0,255),2);
}

void  SaliencyExtractor::computeSaliencyMap()
{
    SM = cv::Mat(Img.size(),Img.type(),cv::Scalar(0,0,255));

    computeSaliency();
    computeSegmentToFixationCorrespondance();


    for(int i=0;i<segments.size();i++)
    {
        SceneSegment S = segments[i];
        cv::rectangle(SM,S.bbox,S.color,CV_FILLED);


        //cv::rectangle(Img,S.bbox,S.color,3);
        //imshow("SalientRegion",Img);

    }
}


void  SaliencyExtractor::computeSaliencyMapAll()
{

    computeSaliency();
//imshow("Saliency",SalImg);
SM = cv::Mat(Img.size(),CV_8UC1,cv::Scalar(0));

        for(int i=0;i<segments.size();i++)
        {
            SceneSegment& S = segments[i];

            if(S.saliency>meanSal & S.area>MIN_AREA)
                SM |= S.mask;

        }

}

void SaliencyExtractor::mergeSegments()
{

    vector<SceneSegment> merged;
    for(int i=0;i<segments.size();i++)
    {
        if(merged.size()<1)
            merged.push_back(segments[i]);
        else
        {
            SceneSegment S = segments[i];
            bool ismerged = false;
            for(int j=0;j<merged.size();j++)
            {
                SceneSegment *Sm = &merged[j];
                Rect overlap = S.bbox & Sm->bbox;
                int intersect_area = overlap.area();
                if(intersect_area > 0.5*S.area | intersect_area > 0.5*Sm->area)
                {
                    int new_x,new_y,new_width,new_height;
                    new_x = min(S.bbox.x,Sm->bbox.x);
                    new_y = min(S.bbox.y,Sm->bbox.y);
                    new_width = max(S.bbox.width,Sm->bbox.width);
                    new_height = max(S.bbox.height,Sm->bbox.height);
                    Rect R(new_x,new_y,new_width,new_height);
                    if(new_width*new_height > 0.7*(Img.cols*Img.rows))   //Very large region
                        continue;
                    Sm->bbox = R;
                    if(S.area < Sm->area)
                        Sm->color = S.color;
                    ismerged = true;
                    break;
                }


            }
            if(!ismerged)
                merged.push_back(S);
        }
    }
//    return merged;
}


void SaliencyExtractor::RGB2Lab(Mat &RGB, vector<Mat> &Lab)
{


        Mat L,a,b;

        L = Mat::zeros(RGB.size(),CV_32FC1);
        a = Mat::zeros(RGB.size(),CV_32FC1);
        b = Mat::zeros(RGB.size(),CV_32FC1);

        for( int y = 0; y < RGB.rows; y++ )
        {
            float* LPtr = L.ptr<float>(y);
            float* aPtr = a.ptr<float>(y);
            float* bPtr = b.ptr<float>(y);
        for( int x = 0; x < RGB.cols; x++ )
        {

            cv::Vec3b bgr = RGB.at<cv::Vec3b>(y,x);
            int sR = bgr[2];
            int sG = bgr[1];
            int sB = bgr[0];

            // sRGB to XYZ conversion


            double R = sR/255.0;
            double G = sG/255.0;
            double B = sB/255.0;

            double r, g, b;

            if(R <= 0.04045)	r = R/12.92;
            else				r = pow((R+0.055)/1.055,2.4);
            if(G <= 0.04045)	g = G/12.92;
            else				g = pow((G+0.055)/1.055,2.4);
            if(B <= 0.04045)	b = B/12.92;
            else				b = pow((B+0.055)/1.055,2.4);

            double X = r*0.4124564 + g*0.3575761 + b*0.1804375;
            double Y = r*0.2126729 + g*0.7151522 + b*0.0721750;
            double Z = r*0.0193339 + g*0.1191920 + b*0.9503041;

            // XYZ to LAB conversion

            double epsilon = 0.008856;
            double kappa   = 903.3;

            /*double Xr = 0.950456;	//whitepoint
            double Yr = 1.0;
            double Zr = 1.088754;	*/

            double Xr = 0.9642;	//whitepoint  (ICC)
            double Yr = 1.0;
            double Zr = 0.8249;

            double xr = X/Xr;
            double yr = Y/Yr;
            double zr = Z/Zr;

            double fx, fy, fz;
            if(xr > epsilon)	fx = pow(xr, 1.0/3.0);
            else				fx = (kappa*xr + 16.0)/116.0;
            if(yr > epsilon)	fy = pow(yr, 1.0/3.0);
            else				fy = (kappa*yr + 16.0)/116.0;
            if(zr > epsilon)	fz = pow(zr, 1.0/3.0);
            else				fz = (kappa*zr + 16.0)/116.0;

            LPtr[x] = 116.0*fy-16.0;
            aPtr[x] = 500.0*(fx-fy);
            bPtr[x] = 200.0*(fy-fz);
        }

    }

        Lab.push_back(L);
        Lab.push_back(a);
        Lab.push_back(b);

}


void SaliencyExtractor::BGR2OCS(Mat &BGR, vector<Mat> &OCS)
{


        Mat O1,O2,O3,O4;

        O1 = Mat::zeros(BGR.size(),CV_32FC1);
        O2 = Mat::zeros(BGR.size(),CV_32FC1);
        O3 = Mat::zeros(BGR.size(),CV_32FC1);
        O4 = Mat::zeros(BGR.size(),CV_32FC1);

        for( int y = 0; y < BGR.rows; y++ )
        {
            float* O1Ptr = O1.ptr<float>(y);
            float* O2Ptr = O2.ptr<float>(y);
            float* O3Ptr = O3.ptr<float>(y);
            float* O4Ptr = O4.ptr<float>(y);
        for( int x = 0; x < BGR.cols; x++ )
        {

            cv::Vec3b bgr = BGR.at<cv::Vec3b>(y,x);
            int sR = bgr[2];
            int sG = bgr[1];
            int sB = bgr[0];

            // sRGB to XYZ conversion


            double R = sR/255.0;
            double G = sG/255.0;
            double B = sB/255.0;

//            R = R<0.04045?R/12.92:pow((R+0.055)/1.055,2.4);
//            G = G<0.04045?G/12.92:pow((G+0.055)/1.055,2.4);
//            B = B<0.04045?B/12.92:pow((B+0.055)/1.055,2.4);
            float nr_c = sqrt(R*R+G*G+B*B)+1e-15;
            R = R/nr_c;
            G = G/nr_c;
            B = B/nr_c;


            double r, g, b;

            if(R <= 0.04045)	r = R/12.92;
            else				r = pow((R+0.055)/1.055,2.4);
            if(G <= 0.04045)	g = G/12.92;
            else				g = pow((G+0.055)/1.055,2.4);
            if(B <= 0.04045)	b = B/12.92;
            else				b = pow((B+0.055)/1.055,2.4);

            double X = r*0.4124564 + g*0.3575761 + b*0.1804375;
            double Y = r*0.2126729 + g*0.7151522 + b*0.0721750;
            double Z = r*0.0193339 + g*0.1191920 + b*0.9503041;


            /*double Xr = 0.950456;	//whitepoint
            double Yr = 1.0;
            double Zr = 1.088754;	*/

            double Xr = 0.9642;	//whitepoint  (ICC)
            double Yr = 1.0;
            double Zr = 0.8249;

            double xr = X/Xr;
            double yr = Y/Yr;
            double zr = Z/Zr;

//            double L = 0.2430*xr + 0.8560*yr - 0.0440*zr;
//            double M = -0.3910*xr + 1.1650*yr + 0.0870*zr;
//            double S = 0.0100*xr - 0.0080*yr + 0.5630*zr;

//            double L = 0.3871*xr + 0.68898*yr - 0.07868*zr;   //HUNT & RLAB
//            double M = -0.22981*xr + 1.18340*yr + 0.04641*zr;
//            double S =  zr;

//            double L = 0.7982*xr + 0.3387*yr - 0.1371*zr;   //CMCCAT2000
//            double M = -0.5918*xr + 1.5512*yr + 0.0406*zr;
//            double S =  0.0008*xr + 0.0239*yr + 0.9753*zr;


            double L = 0.7328*xr + 0.4296*yr - 0.1624*zr;   //CIECAM02
            double M = -0.7036*xr + 1.6975*yr + 0.0061*zr;
            double S =  0.0031*xr + 0.0136*yr + 0.9834*zr;

//            double L = 0.4002*xr + 0.7076*yr - 0.0808*zr;   //Krauskpof
//            double M = -0.2263*xr + 1.1653*yr + 0.0457*zr;
//            double S =  0.9182*zr;

//            double L = 0.2729*xr + 0.6642*yr + 0.0629*zr;   //Christos
//            double M = 0.1002*xr + 0.7876*yr + 0.1122*zr;
//            double S = 0.0178*xr + 0.1096*yr + 0.8726*zr;

            O3Ptr[x] = L-0.5*M-0.5*S;
            O2Ptr[x] = M - 0.5*S - 0.5*L;
            O1Ptr[x] = -0.5*L -0.5*M + S;
            O4Ptr[x] = (L+M+S)/3;

            /*float sum_lms = L+M+S;
            L = L/(sum_lms);
            M = M/(sum_lms);
            S = S/(sum_lms);*/

/*            enum Channel {Long,Medium,Short};
            Channel active_channel,second_active_channel;

            if(L>S)
            {
                if(L>M)
                {
                    active_channel = Long;
                    if(M>S)
                        second_active_channel = Medium;
                    else
                        second_active_channel = Short;
                }
                else
                {
                    active_channel = Medium;
                    second_active_channel = Long;
                }

            }
            else
            {
                if(S>M)
                {
                    active_channel = Short;
                    if(L>M)
                        second_active_channel = Long;
                    else
                        second_active_channel = Medium;
                }
                else
                {
                    active_channel = Medium;
                    second_active_channel = Short;
                }
            }





            float Rstimulus = (active_channel==Long)?L:0;
            float Gstimulus = (active_channel==Medium)?M:0;
            float Bstimulus = (active_channel==Short)?S:0;
            float Ystimulus = ((active_channel==Long&second_active_channel==Medium))?(L+M)/2-S-abs(L-M)/2:0;


            Ystimulus = Ystimulus<0?0:Ystimulus;
            O1Ptr[x] = Bstimulus;
            O2Ptr[x] = Gstimulus;
            O3Ptr[x] = Rstimulus;
            O4Ptr[x] = Ystimulus;*/

            O1Ptr[x] = O1Ptr[x]<0?0:O1Ptr[x];
            O2Ptr[x] = O2Ptr[x]<0?0:O2Ptr[x];
            O3Ptr[x] = O3Ptr[x]<0?0:O3Ptr[x];
            O4Ptr[x] = O4Ptr[x]<0?0:O4Ptr[x];

            nr_c = O1Ptr[x]+O2Ptr[x]+O3Ptr[x]+1e-15;
            O1Ptr[x] = O1Ptr[x]/nr_c;
            O2Ptr[x] = O2Ptr[x]/nr_c;
            O3Ptr[x] = O3Ptr[x]/nr_c;


        }

    }

        OCS.push_back(O1);
        OCS.push_back(O2);
        OCS.push_back(O3);
        OCS.push_back(O4);
}


Mat SaliencyExtractor::ifftShift(Mat &in)
{

    int cx = in.cols/2;
    int cy = in.rows/2;

    Rect Q1(0,0,cx,cy);
    Rect Q2(cx,0,cx,cy);
    Rect Q3(0,cy,cx,cy);
    Rect Q4(cx,cy,cx,cy);

    Mat out=Mat::zeros(in.size(),in.type());
    in(Q1).copyTo(out(Q4));
    in(Q4).copyTo(out(Q1));
    in(Q2).copyTo(out(Q3));
    in(Q3).copyTo(out(Q2));

    return out;
}

void SaliencyExtractor::meshgrid(const cv::Range &xgv, const cv::Range &ygv,
                     cv::Mat &X, cv::Mat &Y)
{
    std::vector<int> t_x, t_y;
    for (int i = xgv.start; i <= xgv.end; i++) t_x.push_back(i);
    for (int i = ygv.start; i <= ygv.end; i++) t_y.push_back(i);
    cv::Mat m_x(t_x); cv::Mat m_y(t_y);
    cv::repeat(m_x.reshape(1,1), m_x.total(), 1, X);
    cv::repeat(m_y.reshape(1,1).t(), 1, m_y.total(), Y);

}


Mat SaliencyExtractor::logGaborMask(int rows, int cols, float sigma, float theta)
{
    cv::Mat X, Y;

    float xm=(round(cols/2)+1);
    float ym=(round(rows/2)+1);
    float xdiv = 1.0/(cols-(cols%2));
    float ydiv = 1.0/(rows-(rows%2));
    int xmin = (1-xm);
    int xmax = (cols-xm);
    int ymin = (1-ym);
    int ymax = (rows-ym);
    cv::Range xr(xmin,xmax),yr(ymin,ymax);
    meshgrid(xr,yr,X,Y);

    X.convertTo(X,CV_32FC1,xdiv);
    Y.convertTo(Y,CV_32FC1,ydiv);

    cv::Mat mask = Mat::ones(Size(rows,cols),CV_32FC1);
    for(int ridx = 0;ridx<rows;ridx++)
    {
        float* XPtr = X.ptr<float>(ridx);
        float* YPtr = Y.ptr<float>(ridx);
        float* maskPtr = mask.ptr<float>(ridx);
        for(int cidx=0;cidx<cols;cidx++)
        {

            if( (XPtr[cidx]*XPtr[cidx] + YPtr[cidx]*YPtr[cidx]) > 0.25)
                maskPtr[cidx] = 0;
        }
    }



    X = X.mul(mask);
    Y = Y.mul(mask);

    X = ifftShift(X);
    Y = ifftShift(Y);

    Mat radius,LG,aux;
    sqrt(X.mul(X) + Y.mul(Y),radius);
    radius.at<float>(1,1) = 1;
    log(radius/theta,aux);

    exp(-aux.mul(aux) / (2 * (sigma*sigma)),LG);
    LG.at<float>(1,1) = 0;

    return LG;
}


Mat SaliencyExtractor::computeGlobalSaliency()
{
    Mat GSal,LG;
    cuda::GpuMat aux,cudaLG,C1Sal,C2Sal,C3Sal,C4Sal;
    //GSal = stimulus_weight[0]*LMS[0]+stimulus_weight[1]*LMS[1]+stimulus_weight[2]*LMS[2]+stimulus_weight[3]*LMS[3];

    LG = logGaborMask(Img.rows,Img.cols,2,0.005);

    Mat planes[] = {LG, LG};
    merge(planes,2,LG);
    cudaLG.upload(LG);

    Mat Lcomplex = Mat(Size(Img.rows,Img.cols),CV_32FC2);
    Mat Lplanes[] = {Mat_<float>(LMS[0]), Mat::zeros(Size(Img.rows,Img.cols),CV_32F)};
    merge(Lplanes,2,Lcomplex);


    gpuImage.upload(Lcomplex);
    cuda::dft(gpuImage,aux,gpuImage.size());
    cuda::multiply(aux,cudaLG,aux);
    cuda::dft(aux,C1Sal,gpuImage.size(),cv::DFT_INVERSE);

    Mat Mcomplex = Mat(Size(Img.rows,Img.cols),CV_32FC2);
    Mat Mplanes[] = {Mat_<float>(LMS[1]), Mat::zeros(Size(Img.rows,Img.cols),CV_32F)};
    merge(Mplanes,2,Mcomplex);
    gpuImage.upload(Mcomplex);
    cuda::dft(gpuImage,aux,gpuImage.size());
    cuda::multiply(aux,cudaLG,aux);
    cuda::dft(aux,C2Sal,gpuImage.size(),cv::DFT_INVERSE);

    Mat Scomplex = Mat(Size(Img.rows,Img.cols),CV_32FC2);
    Mat Splanes[] = {Mat_<float>(LMS[2]), Mat::zeros(Size(Img.rows,Img.cols),CV_32F)};
    merge(Splanes,2,Scomplex);
    gpuImage.upload(Scomplex);
    cuda::dft(gpuImage,aux,gpuImage.size());
    cuda::multiply(aux,cudaLG,aux);
    cuda::dft(aux,C3Sal,gpuImage.size(),cv::DFT_INVERSE);

    Mat Vcomplex = Mat(Size(Img.rows,Img.cols),CV_32FC2);
    Mat Vplanes[] = {Mat_<float>(LMS[3]), Mat::zeros(Size(Img.rows,Img.cols),CV_32F)};
    merge(Vplanes,2,Vcomplex);
    gpuImage.upload(Vcomplex);
    cuda::dft(gpuImage,aux,gpuImage.size());
    cuda::multiply(aux,cudaLG,aux);
    cuda::dft(aux,C4Sal,gpuImage.size(),cv::DFT_INVERSE);


    vector<cuda::GpuMat> gpuMatArray1,gpuMatArray2,gpuMatArray3,gpuMatArray4;
    cuda::split(C1Sal,gpuMatArray1);
    C1Sal = gpuMatArray1[0];

    cuda::split(C2Sal,gpuMatArray2);
    C2Sal = gpuMatArray2[0];

    cuda::split(C3Sal,gpuMatArray3);
    C3Sal = gpuMatArray3[0];


    cuda::split(C4Sal,gpuMatArray4);
    C4Sal = gpuMatArray4[0];

    cuda::multiply(C1Sal,C1Sal,gpuImage);
    cuda::multiply(C2Sal,C2Sal,aux);
    cuda::add(gpuImage,aux,gpuImage);
    cuda::multiply(C3Sal,C3Sal,aux);
    cuda::add(gpuImage,aux,gpuImage);

    cuda::multiply(C4Sal,C4Sal,aux);
    cuda::add(gpuImage,aux,gpuImage);

    gpuImage.download(GSal);

    return GSal;
}


static const double XYZ_to_RGB[3][3] = {
    { 3.24071,	-0.969258,  0.0556352 },
    { -1.53726,	1.87599,    -0.203996 },
    { -0.498571,	0.0415557,  1.05707 }
};

void SaliencyExtractor::cct2rgb(double T, double RGB[3])
{
    int c;
    double xD, yD, X, Y, Z, max;

    if (T <= 4000) {
        xD = 0.27475e9 / (T * T * T) - 0.98598e6 / (T * T) + 1.17444e3 / T + 0.145986;
    } else if (T <= 7000) {
        xD = -4.6070e9 / (T * T * T) + 2.9678e6 / (T * T) + 0.09911e3 / T + 0.244063;
    } else {
        xD = -2.0064e9 / (T * T * T) + 1.9018e6 / (T * T) + 0.24748e3 / T + 0.237040;
    }
    yD = -3 * xD * xD + 2.87 * xD - 0.275;

    //xD = -1.8596e9/(T*T*T) + 1.37686e6/(T*T) + 0.360496e3/T + 0.232632;
    //yD = -2.6046*xD*xD + 2.6106*xD - 0.239156;

    //xD = -1.98883e9/(T*T*T) + 1.45155e6/(T*T) + 0.364774e3/T + 0.231136;
    //yD = -2.35563*xD*xD + 2.39688*xD - 0.196035;

    X = xD / yD;
    Y = 1;
    Z = (1 - xD - yD) / yD;
    max = 0;
    for (c = 0; c < 3; c++) {
        RGB[c] = X * XYZ_to_RGB[0][c] + Y * XYZ_to_RGB[1][c] + Z * XYZ_to_RGB[2][c];
        if (RGB[c] > max) max = RGB[c];
    }
    for (c = 0; c < 3; c++) RGB[c] = RGB[c] / max;
}

double SaliencyExtractor::rgb2cct(int sR,int sG,int sB)
{

    double R = sR/255.0;
    double G = sG/255.0;
    double B = sB/255.0;

    double Tmax, Tmin, testRGB[3];
    Tmin = 1500;
    Tmax = 10000;
    double T = (Tmax + Tmin) / 2;
    while (Tmax - Tmin > 0.1) {
        cct2rgb(T, testRGB);
        if (testRGB[2] / testRGB[0] > B / R)
            Tmax = T;
        else
            Tmin = T;
        T = (Tmax + Tmin) / 2;
    }

    return T;
}
