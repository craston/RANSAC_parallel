#include <stdio.h>
#include <iostream>
#include <stdio.h>
#include <math.h>
#include "opencv2/features2d.hpp"
#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/opencv.hpp"
//#include "SubplotImgs.hpp"

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

void readme();
void sift_detect(Mat img,std::vector<KeyPoint> &kps,  Mat& des );
void findExtrema(std::vector<Mat> dogs,std::vector<KeyPoint> &keypoints , double OctaveID);
void AssignOrientation(std::vector<KeyPoint> &keypoints, std::vector<Mat> blurredImgs);
void getBlurred(Mat img ,std::vector<Mat> &blurredImgs, double k);
void getDoGs(std::vector<Mat> blurredImgs ,std::vector<Mat> &dogs);
bool eliminateEdgeResp(int x , int y, Mat img);
int calculateOrientation( Mat img, Point2f k, double size);
int arg_max(int *x, int len);
void write_kps(Mat img,std::vector<KeyPoint> kps, int i);
/*
 * @function main
 * @brief Main function
 */

#define s 6
#define ROW 3
#define COL 3
#define SIGMA 1.6
#define HIST_ROT_SIZE 36
#define NB_OCTAVES 3

int main( int argc, char** argv )
{

  if( argc != 2 )
  { 
    readme();
    return -1; 
  }

  Mat img = imread( argv[1], IMREAD_GRAYSCALE );

  //img.convertTo(img,CV_32S);
  
  if( !img.data)
  { 
    std::cout<< " --(!) Error reading image " << std::endl; return -1; 
  }
/*---------------------Start of Sift--------------------------------------*/


  std::vector<KeyPoint> keypoints; //store the keypoints of first octave
  Mat descriptors;
  sift_detect(img,keypoints,descriptors);
  cout<<"Keypoints detected: "<<keypoints.size()<<endl;
  cout<<"DescriptorExtractor created with size: "<<descriptors.size()<<endl;


  waitKey(0);
  return 0;
}

/*
 * @function readme
 */

void readme()
{ 
  std::cout << " Usage: ./sift_detector <img1>  IMG missing " << std::endl; 
}
/*
 * @functions SIFT
 */
void sift_detect(Mat img,std::vector<KeyPoint> &kps,  Mat& des)
{
  double i = 1;
  Mat img_copy = img.clone();
  //img.copyTo(img_copy);
  std::vector<Mat> blurredImgs;           //store blurred images of first octave    ------> s+3 =6 
  std::vector<Mat> dogs;            //store difference of gaussian images of first octave ----> s+2 = 5
  std::vector<KeyPoint> keypoints; //store the keypoints of first octave

  for(int j = 0 ; j < NB_OCTAVES ; j++)
  {  
    //initialize the octave with blurring the image
    getBlurred(img,blurredImgs,i);
    //apply difference of gaussian response and get the DoGs
    getDoGs(blurredImgs, dogs);
    //find the keypoints of the current octave
    findExtrema(dogs , keypoints,i);
 
    AssignOrientation(keypoints,blurredImgs);
    cout<<keypoints.size()<<endl;
    kps.insert(kps.end(),keypoints.begin(),keypoints.end());

    //initialize the second octave by downsampling the image with scale of 2*sigma    

    write_kps(img_copy,keypoints,j);
    pyrDown(blurredImgs[5], img, Size( blurredImgs[5].cols/2, blurredImgs[5].rows/2 ) );
    cout<<"img size after downsampling"<<img.size()<<endl;
    pyrDown(img_copy, img_copy, Size( img_copy.cols/2, img_copy.rows/2 ) );

    i += s;
    blurredImgs.clear();
    dogs.clear();
    keypoints.clear();
  }

  //Extract descriptors:
  Ptr<DescriptorExtractor> descriptorExtractor = SIFT::create();
  descriptorExtractor -> compute(img_copy,kps,des);

  return;
}
void getBlurred(Mat img ,std::vector<Mat> &blurredImgs,double idx)
{
  double i = idx;
  double k;
  for (; i < idx + s ; i++)
  {
    Mat out;
    k = pow(2,i/s);
    cout <<"k: "<<k<<endl;
    GaussianBlur( img, out, Size(0,0), k*SIGMA, 0 );
    blurredImgs.push_back(out);
  }
  assert(blurredImgs.size() == s);

  return;
}
void getDoGs(std::vector<Mat> blurredImgs ,std::vector<Mat> &dogs)
{
  for (int i = 1 ; i< blurredImgs.size() ; i++)
  {
    Mat out;
    out = blurredImgs[i] - blurredImgs[i-1];
    dogs.push_back(out);
  }
  return;
}

/*
 * @function findExtrema in DoGs
 */

void findExtrema( std::vector<Mat> dogs ,std::vector<KeyPoint> &keypoints, double OctaveID)
{
  int dogsRows = dogs[0].cols;
  int dogsCols = dogs[0].rows;
  bool ismaxima, isminima, isnon;
  int extremaCount = 0;
  int k, l, kbound, lbound;  //filter indices

  for ( int idx = 1 ; idx < dogs.size()-1 ; idx++ )
  {

    for( int i = 0; i < dogsRows; i++ )
    {
        if (i == 0)
        { k = 0; kbound = 2; }
        else if (i == dogsRows-1)
        { k = -1; kbound = 1; }
        else
        { k = -1; kbound = 2; }

        for( int j = 0; j < dogsCols; j++ )
        {
            if (j == 0)
            { l = 0; lbound = 2; }
            else if (j == dogsCols-1)
            { l =-1; lbound = 1; }
            else
            { l = -1; lbound = 2; }
            
            
            ismaxima = false;
            isminima = false;
            isnon    = false;

            for( ; k < kbound ; k++ )
            {
                for( ; l < lbound; l++ )
                {

                  if(k == 0 && l==0)
                  {
                    if(dogs[idx].at<uchar>(Point(i, j)) > dogs[idx-1].at<uchar>(Point(i + k, j + l)) &&
                       dogs[idx].at<uchar>(Point(i, j)) > dogs[idx+1].at<uchar>(Point(i + k, j + l)))
                          ismaxima = true;

                    else if(dogs[idx].at<uchar>(Point(i, j)) < dogs[idx-1].at<uchar>(Point(i + k, j + l)) &&
                            dogs[idx].at<uchar>(Point(i, j)) < dogs[idx+1].at<uchar>(Point(i + k, j + l)))
                              isminima = true; 

                    else isnon = true;
                  }
                  else
                  {
                    if(dogs[idx].at<uchar>(Point(i, j)) > dogs[idx].at<uchar>(Point(i + k, j + l))&&
                       dogs[idx].at<uchar>(Point(i, j)) > dogs[idx-1].at<uchar>(Point(i + k, j + l)) &&
                       dogs[idx].at<uchar>(Point(i, j)) > dogs[idx+1].at<uchar>(Point(i + k, j + l)))
                          ismaxima = true;

                    else if(dogs[idx].at<uchar>(Point(i, j)) < dogs[idx].at<uchar>(Point(i + k, j + l)) &&
                            dogs[idx].at<uchar>(Point(i, j)) < dogs[idx-1].at<uchar>(Point(i + k, j + l)) &&
                            dogs[idx].at<uchar>(Point(i, j)) < dogs[idx+1].at<uchar>(Point(i + k, j + l)))
                            isminima = true;

                    else isnon = true;
                 } 
                }
                //cout<<"pos1"<<endl;
                //cout<<"here2!"<<endl;
                if(!(ismaxima ^ isminima) || isnon) break;
                //scout<<"pos2"<<endl;
                //cout<<"extremaCount : "<<extremaCount<<endl;

            }
     
            if((ismaxima ^ isminima) && !isnon && eliminateEdgeResp(i,j,dogs[idx]))  //Require one of them to be true only, If both are true the point is neither            
            {  
               KeyPoint K(i,j,pow(2,(idx+OctaveID)/s),-1,0,(int) OctaveID/s,idx); //KeyPoint(x,y,size,angle,response,octave,class_id)
               keypoints.push_back(K);
            }
              
       }
    }
  }
  return;
}

bool eliminateEdgeResp(int x , int y, Mat img)
{
  int dx, dy;
  int dxx, dyy, dxy;
  float TR, DET, r, th;
  float ratioCurv;

  dxx = img.at<uchar>(Point(x + 1, y)) + img.at<uchar>(Point(x - 1, y)) - 2*img.at<uchar>(Point(x, y));
  dyy = img.at<uchar>(Point(x, y + 1)) + img.at<uchar>(Point(x, y - 1)) - 2 * img.at<uchar>(Point(x, y));

  dxy = (img.at<uchar>(Point(x + 1, y + 1)) - img.at<uchar>(Point(x - 1, y + 1)) - img.at<uchar>(Point(x + 1,y - 1)) 
                    + img.at<uchar>(Point(x - 1, y - 1)))/ 2;
  TR  = dxx + dyy;
  DET = dxx*dyy - dxy*dxy;

  //cout <<"Trace, det :"<< TR<< " "<<DET<<endl;

  r = 10;
  th = (r+1)*(r+1)/r;
  ratioCurv = std::abs(TR*TR/DET);
  //cout<< ratioCurv<<endl;
  if(ratioCurv > th || DET < 0) return true;
  else               return false;

}



void AssignOrientation( std::vector<KeyPoint> &keypoints, std::vector<Mat> blurredImgs)
{ 
  double size;
  Point2f p;
  Mat img;
  int j;
  float angle;
  std::vector<KeyPoint> newKeyPoints;
  for(int i = 0 ; i < keypoints.size() ; i++)
  {
    size = keypoints[i].size;
    //cout<<"size: "<<size<<endl;
    p    = keypoints[i].pt;          //keypoint location
    j    = keypoints[i].class_id;    //assigned blurred image index
    img  = blurredImgs[j];            
    angle = calculateOrientation(img,p,size);
    if(angle != -1)
    {
      keypoints[i].angle = angle;
      newKeyPoints.push_back(keypoints[i]);
      cout<<"assigned angle "<<angle<<" to keypoint "<< i << endl;

    }
  }
  keypoints.clear();
  keypoints = newKeyPoints;
  return;
}

int calculateOrientation( Mat img, Point2f p, double size)
{ 
  //Mat gaussian = getGaussianKernel(0,1.5*size);

  //cout<<"Gaussian kernal size "<<gaussian.size()<<endl;
  int region = 8 ,idx=0; //idx for histogram
  int *angle_hist;
  float Lx, Ly,angle,m;

  angle_hist = (int*)malloc(HIST_ROT_SIZE*sizeof(int));
  for (int i = 0 ; i< HIST_ROT_SIZE; i++)
    angle_hist[i] = 0;
  //cout<<"rotation histogram initialized"<<endl;

  if ((p.x < region || p.x >= img.cols - region) || (p.y < region || p.y >= img.rows - region))
    return -1;

  //Apply Gaussian blur of window
  Rect patch(p.x-region/2,p.y-region/2,8,8);
  Mat cropped(img,patch), blurredCrop;
  GaussianBlur(cropped, blurredCrop, Size(0,0),1.5*size);
  if(countNonZero(blurredCrop) < 1)
    cout<<"FAILED blurred ROI is zero"<<endl;
  
  //cout<<"Start angle computation"<<endl;
  //Calculate orientations and place them in histogram according to their magnitude
  for(int i=1; i<region-1;i++)
      for(int j=1; j<region-1;j++)
      {
        //Lx = blurredCrop.at<uchar>(Point(i+1, j)) - blurredCrop.at<uchar>(Point(i-1, j));
        //Ly = blurredCrop.at<uchar>(Point(i, j+1)) - blurredCrop.at<uchar>(Point(i, j-1));
        Lx = cropped.at<uchar>(Point(i+1, j)) - cropped.at<uchar>(Point(i-1, j));
        Ly = cropped.at<uchar>(Point(i, j+1)) - cropped.at<uchar>(Point(i, j-1));
        //cout<<"Lx, Ly done "<<i+j<<endl;
        //cout<<"Lx "<< Lx<< " Ly "<<Ly<<endl;
        if(Lx != 0)
        {
          m = sqrt(pow(Lx,2) + pow(Ly,2));
          angle = atan(Ly/Lx) * 180 / M_PI;

          if(angle < 0)
            angle = angle + 360;

          idx = (int) (angle/10);
          //cout<<"angle "<<angle<<endl;
          //cout<<"index of bin:"<<idx<<endl;
          angle_hist[idx] += m; 
        }
      }

      //cout<<angle<<" "<<m<<endl;
      //cout<<"orientation found"<<endl;
      angle = arg_max(angle_hist, HIST_ROT_SIZE) * 10;
  //CALC MAGNITUDE, ORIENTATIONS, PEAKS

  return angle;
}


int arg_max(int *x, int len){
  int max_idx = 0;
  for (int i = 1 ; i< len ; i++)
  {
    if(x[i] > x[max_idx])
      max_idx = i;
  }
  return max_idx;
}

void write_kps(Mat img,std::vector<KeyPoint> kps, int i)
{
  cout<<"number of keypoints in octave "<<i<<": "<<kps.size()<<endl;
  Mat img_keypoints;
  drawKeypoints( img, kps, img_keypoints,  Scalar(255,0,0), DrawMatchesFlags::DEFAULT);
    if(i==0)
      imwrite("./first_keypoints.jpg",img_keypoints);
    else if (i==1)
      imwrite("./second_keypoints.jpg",img_keypoints);
    else if (i==2)
      imwrite("./third_keypoints.jpg",img_keypoints);
    else if (i==3)
      imwrite("./fourth_keypoints.jpg",img_keypoints);
    else if (i==4)
      imwrite("./fifth_keypoints.jpg",img_keypoints);
}

