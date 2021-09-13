/*

Class Name: Utility functions
Author Name: J. Rafid S.
Author URI: www.azaditech.com
Relevant Publication: Siddiqui, J. R. , Andreasson, H. , Driankov, D. & Lilienthal, A. J. (2016). Towards visual mapping in industrial environments: a heterogeneous task-specific and saliency driven approach. IEEE International Conference on Robotics and Automation (ICRA). , Stockholm, Sweden, May, 2016.

*/

#ifndef UTILS_H
#define UTILS_H

#include<iostream>
#include<eigen3/Eigen/Dense>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <fstream>


static void RGB2HSV(float r, float g, float b,
                    float &h, float &s, float &v)
{
    float rgb_max = std::max(r, std::max(g, b));
    float rgb_min = std::min(r, std::min(g, b));
    float delta = rgb_max - rgb_min;
    s = delta / (rgb_max + 1e-20f);
    v = rgb_max;

    float hue;
    if (r == rgb_max)
        hue = (g - b) / (delta + 1e-20f);
    else if (g == rgb_max)
        hue = 2 + (b - r) / (delta + 1e-20f);
    else
        hue = 4 + (r - g) / (delta + 1e-20f);
    if (hue < 0)
        hue += 6.f;
    h = hue * (1.f / 6.f);
}

void HSV2RGB(int Hi, int Si, int Vi, int &R,int &G, int &B) {

    double h,s,v;
    h = Hi / 255.0;
    s = Si / 255.0;
    v = Vi / 255.0;

    double r, g, b;

    int i = int(h * 6);
    double f = h * 6 - i;
    double p = v * (1 - s);
    double q = v * (1 - f * s);
    double t = v * (1 - (1 - f) * s);

    switch(i % 6){
        case 0: r = v, g = t, b = p; break;
        case 1: r = q, g = v, b = p; break;
        case 2: r = p, g = v, b = t; break;
        case 3: r = p, g = q, b = v; break;
        case 4: r = t, g = p, b = v; break;
        case 5: r = v, g = p, b = q; break;
    }

    R = r * 255;
    G = g * 255;
    B = b * 255;
}

void getColors(std::vector<cv::Vec3f>& colors,int numRegions)
{
    colors.clear();
    for (int k = 0; k < numRegions + 2; ++k) {
        cv::Vec3f color;
        color[0] = (rand() % 256)/255.0+0.1;
        color[1] = (rand() % 256)/255.0+0.1;
        color[2] = (rand() % 256)/255.0+0.1;
        colors.push_back(color);
    }
}

float histDistMaxPooling(Eigen::VectorXd P,Eigen::VectorXd Q)
{
    int r1,r2;

    P.maxCoeff(&r1);
    Q.maxCoeff(&r2);
    float dist = abs(r1-r2);
    dist = dist/P.size();
    return dist;
}

float histDistMaxPooling(Eigen::MatrixXd P,Eigen::MatrixXd Q)
{
    int r1,c1,r2,c2;
    Eigen::VectorXd sum1,sum2;
    sum1 = P.rowwise().sum();
    sum2 = Q.rowwise().sum();
    sum1.maxCoeff(&r1);
    sum2.maxCoeff(&r2);

    sum1 = P.colwise().sum();
    sum2 = Q.colwise().sum();
    sum1.maxCoeff(&c1);
    sum2.maxCoeff(&c2);

    float dist = (abs(r1-r2)+abs(c1-c2))/2;
    dist = dist/P.size();

    return dist;
}

float histDistCorrelation(Eigen::MatrixXd P,Eigen::MatrixXd Q)
{
    Eigen::MatrixXd norm_hist1 = P - ((P.sum()/P.size())*(Eigen::MatrixXd::Ones(P.size(),P.size())));
    Eigen::MatrixXd norm_hist2 = Q - ((Q.sum()/P.size())*(Eigen::MatrixXd::Ones(Q.size(),Q.size())));
    float dissim = norm_hist1.cwiseProduct(norm_hist1).sum() * norm_hist2.cwiseProduct(norm_hist2).sum() ;
    float dist = norm_hist1.cwiseProduct(norm_hist2).sum()/sqrt(dissim);
    return dist;
}

float histDistChiSquare(Eigen::VectorXd P,Eigen::VectorXd Q)
{
       Eigen::MatrixXd dissim = P - Q;
       dissim = dissim.cwiseProduct(dissim);
       float dist = dissim.cwiseQuotient(P).sum();
   return dist;
}


float histDistQuadraticChi(Eigen::VectorXd P,Eigen::VectorXd Q,float m)
{

    Eigen::VectorXd aux,Z,D;
    Eigen::Matrix<bool,Eigen::Dynamic,Eigen::Dynamic> F;
    aux.setOnes();
    Eigen::MatrixXd A(P.size(),Q.size());
    for(int i=0;i<P.size();i++)
        A.row(i)=(P[i]*aux)-Q;

    Z = A*(P+Q);

    F = Z.array()==0;
    Z = Z.cwiseProduct(F.cast<double>());
    Z = Z.array().pow(m);
    D = (P-Q).cwiseQuotient(Z);

    float dist = sqrt((D.transpose()*A*D));

    return dist;
}

float getLikelihood(Eigen::VectorXd obs, Eigen::VectorXd mu,Eigen::MatrixXd S)
{

    Eigen::Matrix3d invS(Eigen::Matrix3d::Zero());
    if(S.determinant()!=0)
        invS = S.inverse();
    else
        return 0.5;


    float L;
    float n = obs.size()/2.0;
    Eigen::VectorXd D = obs-mu;
    L = D.transpose()*invS*D;
    L = (1/(pow(2*M_PI,n)*sqrt(S.determinant()))) * exp(-0.5*L);

    return L;
}


double deg2Rad(const double deg)
{
    return (deg * (M_PI / 180.0));
}

double rad2Deg(const double rad)
{
    return ((180.0 / M_PI) * rad);
}

struct LAB
{

    double l;

    double a;

    double b;
    LAB(double _l,double _a,double _b)
    {
        l = _l;
        a = _a;
        b = _b;
    }
};

double CIEDE2000(const LAB &lab1,const LAB &lab2)
{

    const double k_L = 1.0, k_C = 1.0, k_H = 1.0;
    const double deg360InRad = deg2Rad(360.0);
    const double deg180InRad = deg2Rad(180.0);
    const double pow25To7 = 6103515625.0;

    /*
     * Step 1
     */
    /* Equation 2 */
    double C1 = sqrt((lab1.a * lab1.a) + (lab1.b * lab1.b));
    double C2 = sqrt((lab2.a * lab2.a) + (lab2.b * lab2.b));
    /* Equation 3 */
    double barC = (C1 + C2) / 2.0;
    /* Equation 4 */
    double G = 0.5 * (1 - sqrt(pow(barC, 7) / (pow(barC, 7) + pow25To7)));
    /* Equation 5 */
    double a1Prime = (1.0 + G) * lab1.a;
    double a2Prime = (1.0 + G) * lab2.a;
    /* Equation 6 */
    double CPrime1 = sqrt((a1Prime * a1Prime) + (lab1.b * lab1.b));
    double CPrime2 = sqrt((a2Prime * a2Prime) + (lab2.b * lab2.b));
    /* Equation 7 */
    double hPrime1;
    if (lab1.b == 0 && a1Prime == 0)
        hPrime1 = 0.0;
    else {
        hPrime1 = atan2(lab1.b, a1Prime);
        /*
         * This must be converted to a hue angle in degrees between 0
         * and 360 by addition of 2􏰏 to negative hue angles.
         */
        if (hPrime1 < 0)
            hPrime1 += deg360InRad;
    }
    double hPrime2;
    if (lab2.b == 0 && a2Prime == 0)
        hPrime2 = 0.0;
    else {
        hPrime2 = atan2(lab2.b, a2Prime);
        /*
         * This must be converted to a hue angle in degrees between 0
         * and 360 by addition of 2􏰏 to negative hue angles.
         */
        if (hPrime2 < 0)
            hPrime2 += deg360InRad;
    }

    /*
     * Step 2
     */
    /* Equation 8 */
    double deltaLPrime = lab2.l - lab1.l;
    /* Equation 9 */
    double deltaCPrime = CPrime2 - CPrime1;
    /* Equation 10 */
    double deltahPrime;
    double CPrimeProduct = CPrime1 * CPrime2;
    if (CPrimeProduct == 0)
        deltahPrime = 0;
    else {
        /* Avoid the fabs() call */
        deltahPrime = hPrime2 - hPrime1;
        if (deltahPrime < -deg180InRad)
            deltahPrime += deg360InRad;
        else if (deltahPrime > deg180InRad)
            deltahPrime -= deg360InRad;
    }
    /* Equation 11 */
    double deltaHPrime = 2.0 * sqrt(CPrimeProduct) *
        sin(deltahPrime / 2.0);

    /*
     * Step 3
     */
    /* Equation 12 */
    double barLPrime = (lab1.l + lab2.l) / 2.0;
    /* Equation 13 */
    double barCPrime = (CPrime1 + CPrime2) / 2.0;
    /* Equation 14 */
    double barhPrime, hPrimeSum = hPrime1 + hPrime2;
    if (CPrime1 * CPrime2 == 0) {
        barhPrime = hPrimeSum;
    } else {
        if (fabs(hPrime1 - hPrime2) <= deg180InRad)
            barhPrime = hPrimeSum / 2.0;
        else {
            if (hPrimeSum < deg360InRad)
                barhPrime = (hPrimeSum + deg360InRad) / 2.0;
            else
                barhPrime = (hPrimeSum - deg360InRad) / 2.0;
        }
    }
    /* Equation 15 */
    double T = 1.0 - (0.17 * cos(barhPrime - deg2Rad(30.0))) +
        (0.24 * cos(2.0 * barhPrime)) +
        (0.32 * cos((3.0 * barhPrime) + deg2Rad(6.0))) -
        (0.20 * cos((4.0 * barhPrime) - deg2Rad(63.0)));
    /* Equation 16 */
    double deltaTheta = deg2Rad(30.0) *
        exp(-pow((barhPrime - deg2Rad(275.0)) / deg2Rad(25.0), 2.0));
    /* Equation 17 */
    double R_C = 2.0 * sqrt(pow(barCPrime, 7.0) /
        (pow(barCPrime, 7.0) + pow25To7));
    /* Equation 18 */
    double S_L = 1 + ((0.015 * pow(barLPrime - 50.0, 2.0)) /
        sqrt(20 + pow(barLPrime - 50.0, 2.0)));
    /* Equation 19 */
    double S_C = 1 + (0.045 * barCPrime);
    /* Equation 20 */
    double S_H = 1 + (0.015 * barCPrime * T);
    /* Equation 21 */
    double R_T = (-sin(2.0 * deltaTheta)) * R_C;

    /* Equation 22 */
    double deltaE = sqrt(
        pow(deltaLPrime / (k_L * S_L), 2.0) +
        pow(deltaCPrime / (k_C * S_C), 2.0) +
        pow(deltaHPrime / (k_H * S_H), 2.0) +
        (R_T * (deltaCPrime / (k_C * S_C)) * (deltaHPrime / (k_H * S_H))));

    return (deltaE);
}

cv::Mat lookupTable(int levels) {
    int factor = 256 / levels;
    cv::Mat table(1, 256, CV_8U);
    uchar *p = table.data;

    for(int i = 0; i < 128; ++i) {
        p[i] = factor * (i / factor);
    }

    for(int i = 128; i < 256; ++i) {
        p[i] = factor * (1 + (i / factor)) - 1;
    }

    return table;
}

cv::Mat colorReduce(const cv::Mat &image, int levels) {
    cv::Mat table = lookupTable(levels);

    std::vector<cv::Mat> c;
    cv::split(image, c);
    for (std::vector<cv::Mat>::iterator i = c.begin(), n = c.end(); i != n; ++i) {
        cv::Mat &channel = *i;
        cv::LUT(channel.clone(), table, channel);
    }

    cv::Mat reduced;
    cv::merge(c, reduced);
    return reduced;
}

namespace Eigen{
template<class Matrix>
void write_binary(const char* filename, const Matrix& matrix){
    std::ofstream out(filename,std::ios::out | std::ios::binary | std::ios::trunc);
    int rows=matrix.rows(), cols=matrix.cols();
    out.write((char*) (&rows), sizeof(int));
    out.write((char*) (&cols), sizeof(int));
    out.write((char*) matrix.data(), rows*cols*sizeof(typename Matrix::Scalar) );
    out.close();
}
template<class Matrix>
void read_binary(const char* filename, Matrix& matrix){
    std::ifstream in(filename,std::ios::in | std::ios::binary);
    int rows=0, cols=0;
    in.read((char*) (&rows),sizeof(int));
    in.read((char*) (&cols),sizeof(int));
    matrix.resize(rows, cols);
    in.read( (char *) matrix.data() , rows*cols*sizeof(typename Matrix::Scalar) );
    in.close();
}
}


void rgb2xyz(int R, int G, int B, int* xyz) {

        float rf, gf, bf;

        float r, g, b, X, Y, Z;



        r = R/255.0f; //R 0..1

        g = G/255.0f; //G 0..1

        b = B/255.0f; //B 0..1



        if (r <= 0.04045)

            r = r/12;

        else

            r = (float) pow((r+0.055)/1.055,2.4);



        if (g <= 0.04045)

            g = g/12;

        else

            g = (float) pow((g+0.055)/1.055,2.4);



        if (b <= 0.04045)

            b = b/12;

        else

            b = (float) pow((b+0.055)/1.055,2.4);



        X =  0.436052025f*r     + 0.385081593f*g + 0.143087414f *b;

        Y =  0.222491598f*r     + 0.71688606f *g + 0.060621486f *b;

        Z =  0.013929122f*r     + 0.097097002f*g + 0.71418547f  *b;



        xyz[1] = (int) (255*Y + .5);

        xyz[0] = (int) (255*X + .5);

        xyz[2] = (int) (255*Z + .5);

    }


void rgb2lab(int R, int G, int B, int* lab) {




        float r, g, b, X, Y, Z, fx, fy, fz, xr, yr, zr;

        float Ls, as, bs;

        float eps = 216.f/24389.f;

        float k = 24389.f/27.f;



        float Xr = 0.964221f;  // reference white D50

        float Yr = 1.0f;

        float Zr = 0.825211f;



        // RGB to XYZ

        r = R/255.0f; //R 0..1

        g = G/255.0f; //G 0..1

        b = B/255.0f; //B 0..1



        // assuming sRGB (D65)

        if (r <= 0.04045)

            r = r/12;

        else

            r = (float) pow((r+0.055)/1.055,2.4);



        if (g <= 0.04045)

            g = g/12;

        else

            g = (float) pow((g+0.055)/1.055,2.4);



        if (b <= 0.04045)

            b = b/12;

        else

            b = (float) pow((b+0.055)/1.055,2.4);





        X =  0.436052025f*r     + 0.385081593f*g + 0.143087414f *b;

        Y =  0.222491598f*r     + 0.71688606f *g + 0.060621486f *b;

        Z =  0.013929122f*r     + 0.097097002f*g + 0.71418547f  *b;



        // XYZ to Lab

        xr = X/Xr;

        yr = Y/Yr;

        zr = Z/Zr;



        if ( xr > eps )

            fx =  (float) pow(xr, 1/3.);

        else

            fx = (float) ((k * xr + 16.) / 116.);



        if ( yr > eps )

            fy =  (float) pow(yr, 1/3.);

        else

        fy = (float) ((k * yr + 16.) / 116.);



        if ( zr > eps )

            fz =  (float) pow(zr, 1/3.);

        else

            fz = (float) ((k * zr + 16.) / 116);



        Ls = ( 116 * fy ) - 16;

        as = 500*(fx-fy);

        bs = 200*(fy-fz);



        lab[0] = (int) (2.55*Ls + .5);

        lab[1] = (int) (as + .5);

        lab[2] = (int) (bs + .5);

    }
#endif // UTILS_H
