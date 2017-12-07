#include <stdio.h>
#include <iostream>
#include <stdio.h>
#include <iostream>
#include <math.h>
#include "opencv2/features2d.hpp"
#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/opencv.hpp"
#include "SubplotImgs.hpp"

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

void readme();
void findExtrema( std::vector<Mat> dogs,std::vector<KeyPoint> &keypoints );
void AssignOrientation( std::vector<KeyPoint> &keypoints);
void getBlurred(Mat img ,std::vector<Mat> &blurredImgs, double k);
void getDoGs(std::vector<Mat> blurredImgs ,std::vector<Mat> &dogs);
bool eliminateEdgeResp(int x , int y, Mat img);
/*
 * @function main
 * @brief Main function
 */

#define s 6
#define ROW 3
#define COL 3




int main( int argc, char** argv )
{

  if( argc != 2 )
  { 
    readme();
    return -1; 
  }

  Mat img = imread( argv[1], IMREAD_GRAYSCALE );
  if( !img.data)
  { 
    std::cout<< " --(!) Error reading image " << std::endl; return -1; 
  }
/*---------------------Start of Sift--------------------------------------*/


  std::vector<Mat> blurredImgs;           //store blurred images of first octave    ------> s+3 =6 
  std::vector<Mat> dogs;            //store difference of gaussian images of first octave ----> s+2 = 5
  std::vector<KeyPoint> keypoints; //store the keypoints of first octave
  //std::vector<Mat> NMSdogs;     //store features extraced from DoG      ------> s =3 

  //initialize the first octave with blurring the image
  double k= sqrt(2);
  double i = 1;
  getBlurred(img,blurredImgs,i);
  //apply difference of gaussian response and get the DoGs
  getDoGs(blurredImgs, dogs);
  //find the keypoints of the current octave
  findExtrema(dogs , keypoints); 


  //initialize the second octave by downsampling the image with scale of 2*sigma
  Mat img_downSampledOnce;
  pyrDown(blurredImgs[5], img_downSampledOnce, Size( blurredImgs[5].cols/2, blurredImgs[5].rows/2 ) );
  
  std::vector<Mat> blurredImgs1;           //store blurred images of second octave    ------> s+3 =6 
  std::vector<Mat> dogs1;            //store difference of gaussian images of second octave ----> s+2 = 5
  std::vector<KeyPoint> keypoints1;

  k = pow(sqrt(2),s);
  i += s;
  getBlurred(img_downSampledOnce,blurredImgs1,i);
  //apply difference of gaussian response and get the DoGs
  getDoGs(blurredImgs1, dogs1);
  //find the keypoints of the current octave
  findExtrema(dogs1 , keypoints1); 
  //imshow("downSampled image",img_downSampledOnce);
  
  Mat img_downSampledTwice;
  pyrDown(blurredImgs1[5], img_downSampledTwice, Size( blurredImgs1[5].cols/2, blurredImgs1[5].rows/2 ) );
  //imshow("downSampled image",img_downSampledTwice);
  
  std::vector<Mat> blurredImgs2;           //store blurred images of third octave    ------> s+3 =6 
  std::vector<Mat> dogs2;            //store difference of gaussian images of third octave ----> s+2 = 5
  std::vector<KeyPoint> keypoints2;

  k = pow(sqrt(2),s);
  i += s;
  getBlurred(img_downSampledTwice,blurredImgs2,i);
  //apply difference of gaussian response and get the DoGs
  getDoGs(blurredImgs2, dogs2);
  //find the keypoints of the current octave
  findExtrema(dogs2 , keypoints2); 
  
  //Draw keypoints of all scales on its images
  Mat img_keypoints;
  drawKeypoints( img, keypoints, img_keypoints, Scalar(255,0,0), DrawMatchesFlags::DEFAULT );

  Mat img_keypoints1;
  drawKeypoints( img_downSampledOnce, keypoints1, img_keypoints1,  Scalar(255,0,0), DrawMatchesFlags::DEFAULT ) ;
  
  Mat img_keypoints2;
  drawKeypoints( img_downSampledTwice, keypoints2, img_keypoints2,  Scalar(255,0,0), DrawMatchesFlags::DEFAULT ) ;
  


  //PLOTTING SECTION
  //Move all images to one image for better view and plot
  Mat blurredSubplot = SubPlot(blurredImgs);
  Mat DogSubplot = SubPlot(dogs);
  //Mat NmsSubplot = SubPlot(NMSdogs);
  
  //Mat blurredSubplot1 = SubPlot(blurredImgs1,0);

  //imshow("DOG Images", DogSubplot);
  //imshow("Blurred Images", blurredSubplot);
  //imshow("Blurred Images third octave", blurredImgs2[2]);
  //imshow("Extrema Images of first octave",img_keypoints);
  imwrite("./first_keypoints.jpg",img_keypoints);
  imwrite("./second_keypoints.jpg",img_keypoints1);
  imwrite("./third_keypoints.jpg",img_keypoints2);

  cout<<"number of keypoints in the first octave  "<<keypoints.size()<<endl;
  cout<<"number of keypoints in the second octave "<<keypoints1.size()<<endl;
  cout<<"number of keypoints in the third octave  "<<keypoints2.size()<<endl;
  //imshow("Extrema Images of second octave",img_keypoints1);
  //imshow("Extrema Images of third octave",img_keypoints2);


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
void getBlurred(Mat img ,std::vector<Mat> &blurredImgs,double idx)
{
  double i = idx;
  double k;
  for (; i < idx + 6 ; i++)
  {
    Mat out;
    //k = pow(2,i/s);
    k = pow(2,i/s);
    cout <<"k: "<<k<<endl;
    GaussianBlur( img, out, Size(0,0), k*1.6, 0 );
    k = pow(2,i/s);
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

void findExtrema( std::vector<Mat> dogs ,std::vector<KeyPoint> &keypoints)
{
  int dogsRows = dogs[0].rows;
  int dogsCols = dogs[0].cols;
  //std::vector<Mat> nmsDogs;
  bool ismaxima, isminima;
  int k, l, kbound, lbound;  //filter indices

  for ( int idx = 1 ; idx < dogs.size() -1 ; idx ++ )
  {
    //Mat fill = Mat::zeros(dogs[0].size(), CV_8UC1);

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
            
            for( ; k < kbound ; k++ )
            {
                for(; l < lbound; l++ )
                {
                //Compare each dog image idx with image idx -1 and idx +1 and its neighbor
                  if(k == 0 && l==0)
                  {
                    if(dogs[idx].at<uchar>(Point(i, j)) > dogs[idx-1].at<uchar>(Point(i + k, j + l)) &&
                       dogs[idx].at<uchar>(Point(i, j)) > dogs[idx+1].at<uchar>(Point(i + k, j + l)))
                          ismaxima = true;

                    else if(dogs[idx].at<uchar>(Point(i, j)) < dogs[idx-1].at<uchar>(Point(i + k, j + l)) &&
                            dogs[idx].at<uchar>(Point(i, j)) < dogs[idx+1].at<uchar>(Point(i + k, j + l)))
                               isminima = true;    
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
                 } 
                }
                if(!(ismaxima ^ isminima)) break;
            }
            if((ismaxima ^ isminima))// && eliminateEdgeResp(i,j,dogs[idx]))  //Require one of them to be true only, If both are true the point is neither
            {  //fill.at<uchar>(Point(i, j)) = 255; 
               KeyPoint K(i,j,pow(sqrt(2),i),-1,0,idx,-1); //KeyPoint(x,y,size,angle,response,octave,class_id)
               keypoints.push_back(K);
            }
              
       }
    }
    //nmsDogs.push_back(fill);
  }
  return;//nmsDogs;
}

bool eliminateEdgeResp(int x , int y, Mat img)
{
  int dx, dy;
  int dxx, dyy, dxy;
  float TR, DET, r, th;
  float ratioCurv;
  /*
  dx = Scalar(img.at<uchar>(Point(x-1, y-1))).val[0] - Scalar(img.at<uchar>(Point(x-1, y+1))).val[0] +
       2*Scalar(img.at<uchar>(Point(x, y-1))).val[0] - 2*Scalar(img.at<uchar>(Point(x, y+1))).val[0] +
       Scalar(img.at<uchar>(Point(x+1, y-1))).val[0] - Scalar(img.at<uchar>(Point(x+1, y+1))).val[0];
  
  dy = Scalar(img.at<uchar>(Point(x-1, y-1))).val[0] - Scalar(img.at<uchar>(Point(x+1, y-1))).val[0] +
       2*Scalar(img.at<uchar>(Point(x-1, y))).val[0] - 2*Scalar(img.at<uchar>(Point(x+1, y))).val[0] +
       Scalar(img.at<uchar>(Point(x-1, y+1))).val[0] - Scalar(img.at<uchar>(Point(x+1, y+1))).val[0];
  
  dxx = dx*dx;
  dyy = dy*dy;
  dxy = dx*dy;
*/
  dxx = img.at<uchar>(Point(x + 1, y)) + img.at<uchar>(Point(x - 1, y)) - 2*img.at<uchar>(Point(x, y));
  dyy = img.at<uchar>(Point(x, y + 1)) + img.at<uchar>(Point(x, y - 1)) - 2 * img.at<uchar>(Point(x, y));
  //dss = img[2](p.x, p.y) + img[0](p.x, p.y) - 2 * img[1](p.x, p.y);
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



void AssignOrientation( std::vector<KeyPoint> &keypoints)
{

}


/*
Scalar p = img.at<uchar>(Point(i, j));                   //Returns a vector of values 
Scalar(img.at<uchar>(Point(i, j))).val[0]; || p.val[0]  //Returnd a value of GrayScale pixel
fill.at<uchar>(Point(i, j)) = color;*/                 //Assign a color to point i,j
/*double min , max ; 
cv::minMaxLoc(dogs[2],&min,&max);
*/