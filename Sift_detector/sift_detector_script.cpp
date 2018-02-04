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
void findExtrema( std::vector<Mat> dogs,std::vector<KeyPoint> &keypoints , double OctaveID);
void AssignOrientation( std::vector<KeyPoint> &keypoints, std::vector<Mat> blurredImgs);
void getBlurred(Mat img ,std::vector<Mat> &blurredImgs, double k);
void getDoGs(std::vector<Mat> blurredImgs ,std::vector<Mat> &dogs);
bool eliminateEdgeResp(int x , int y, Mat img);
int calculateOrientation( Mat img, Point2f k, double size);
int arg_max(int *x, int len);
/*
 * @function main
 * @brief Main function
 */

#define s 6
#define ROW 3
#define COL 3
#define SIGMA 1.6
#define HIST_ROT_SIZE 36
 

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


  std::vector<Mat> blurredImgs;           //store blurred images of first octave    ------> s+3 =6 
  std::vector<Mat> dogs;            //store difference of gaussian images of first octave ----> s+2 = 5
  std::vector<KeyPoint> keypoints; //store the keypoints of first octave
  //std::vector<Mat> NMSdogs;     //store features extraced from DoG      ------> s =3 

  //initialize the first octave with blurring the image
  double i = 1;
  getBlurred(img,blurredImgs,i);
  //apply difference of gaussian response and get the DoGs
  getDoGs(blurredImgs, dogs);
  //find the keypoints of the current octave
  findExtrema(dogs , keypoints,i); 
  AssignOrientation(keypoints,blurredImgs);

  //initialize the second octave by downsampling the image with scale of 2*sigma
  Mat img_downSampledOnce;
  pyrDown(blurredImgs[5], img_downSampledOnce, Size( blurredImgs[5].cols/2, blurredImgs[5].rows/2 ) );
  /*
  Mat descriptors;
  //SurfDescriptorExtractor ext;
  Ptr<DescriptorExtractor> descriptorExtractor = SIFT::create();
  descriptorExtractor -> compute(blurredImgs[0],keypoints,descriptors);
  cout<<"DescriptorExtractor created with size: "<<descriptors.size()<<endl;
*/




  std::vector<Mat> blurredImgs1;           //store blurred images of second octave    ------> s+3 =6 
  std::vector<Mat> dogs1;            //store difference of gaussian images of second octave ----> s+2 = 5
  std::vector<KeyPoint> keypoints1;

  i += s;
  getBlurred(img_downSampledOnce,blurredImgs1,i);
  //apply difference of gaussian response and get the DoGs
  getDoGs(blurredImgs1, dogs1);
  //find the keypoints of the current octave
  findExtrema(dogs1 , keypoints1 , i); 
  //imshow("downSampled image",img_downSampledOnce);
  AssignOrientation(keypoints1,blurredImgs1);

  Mat img_downSampledTwice;
  pyrDown(blurredImgs1[5], img_downSampledTwice, Size( blurredImgs1[5].cols/2, blurredImgs1[5].rows/2 ) );
  //imshow("downSampled image",img_downSampledTwice);
  
  std::vector<Mat> blurredImgs2;           //store blurred images of third octave    ------> s+3 =6 
  std::vector<Mat> dogs2;            //store difference of gaussian images of third octave ----> s+2 = 5
  std::vector<KeyPoint> keypoints2;

  i += s;
  getBlurred(img_downSampledTwice,blurredImgs2,i);
  //apply difference of gaussian response and get the DoGs
  getDoGs(blurredImgs2, dogs2);
  //find the keypoints of the current octave
  findExtrema(dogs2 , keypoints2, i); 
  AssignOrientation(keypoints2,blurredImgs2);

  //CAT keypoints
  keypoints.insert(keypoints.end(),keypoints1.begin(),keypoints1.end());
  keypoints.insert(keypoints.end(),keypoints2.begin(),keypoints2.end());
  //find_descriptors
  Mat descriptors;
  //SurfDescriptorExtractor ext;
  Ptr<DescriptorExtractor> descriptorExtractor = SIFT::create();
  descriptorExtractor -> compute(img,keypoints,descriptors);
  cout<<"DescriptorExtractor created with size: "<<descriptors.size()<<endl;


  //Draw keypoints of all scales on its images
  Mat img_keypoints;
  drawKeypoints( img, keypoints, img_keypoints, Scalar(255,0,0), DrawMatchesFlags::DEFAULT );

  Mat img_keypoints1;
  drawKeypoints( img_downSampledOnce, keypoints1, img_keypoints1,  Scalar(255,0,0), DrawMatchesFlags::DEFAULT ) ;
  
  Mat img_keypoints2;
  drawKeypoints( img_downSampledTwice, keypoints2, img_keypoints2,  Scalar(255,0,0), DrawMatchesFlags::DEFAULT ) ;
  


  //PLOTTING SECTION
  //Move all images to one image for better view and plot
  //Mat blurredSubplot = SubPlot(blurredImgs);
  //Mat DogSubplot = SubPlot(dogs);
  //Mat NmsSubplot = SubPlot(NMSdogs);
  
  //Mat blurredSubplot1 = SubPlot(blurredImgs1,0);

  //imshow("DOG Images", DogSubplot);
  //imshow("Blurred Images", blurredSubplot);
  //imshow("Blurred Images third octave", blurredImgs2[2]);
  //imshow("Extrema Images of first octave",img_keypoints);
  imwrite("./first_keypoints.jpg",img_keypoints);
  imwrite("./second_keypoints.jpg",img_keypoints1);
  imwrite("./third_keypoints.jpg",img_keypoints2);

  cout<<"number of keypoints in the first octave  "<<keypoints.size()-keypoints1.size()-keypoints2.size()<<endl;
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
  for (; i < idx + s ; i++)
  {
    Mat out;
    //k = pow(2,i/s);
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
    //cv::subtract(blurredImgs[i], blurredImgs[i-1], out, cv::noArray(), CV_32S);
    dogs.push_back(out);
    /*for(int i = 0 ; i<out.rows ; i++)
      for (int j = 0; j < out.cols ; j++)
      {
        if(out.at<char>(Point(i, j))<0)
          cout<<"neg FOUND!!!"<<endl;
      }*/
  }
  return;
}

/*
 * @function findExtrema in DoGs
 */

void findExtrema( std::vector<Mat> dogs ,std::vector<KeyPoint> &keypoints, double OctaveID)
{
  int dogsRows = dogs[0].rows;
  int dogsCols = dogs[0].cols;
  //std::vector<Mat> nmsDogs;
  bool ismaxima, isminima, isnon;
  int extremaCount;
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
            isnon    = false;
            extremaCount = 0;
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
                          ///{extremaCount+=1;cout<<"here"<<endl;}
                    else if(dogs[idx].at<uchar>(Point(i, j)) < dogs[idx-1].at<uchar>(Point(i + k, j + l)) &&
                            dogs[idx].at<uchar>(Point(i, j)) < dogs[idx+1].at<uchar>(Point(i + k, j + l)))
                              isminima = true;    
                               //extremaCount+=-1;
                    else isnon = true;
                  }
                  else
                  {
                    if(dogs[idx].at<uchar>(Point(i, j)) > dogs[idx].at<uchar>(Point(i + k, j + l))&&
                       dogs[idx].at<uchar>(Point(i, j)) > dogs[idx-1].at<uchar>(Point(i + k, j + l)) &&
                       dogs[idx].at<uchar>(Point(i, j)) > dogs[idx+1].at<uchar>(Point(i + k, j + l)))
                          //extremaCount+=1;
                          ismaxima = true;

                    else if(dogs[idx].at<uchar>(Point(i, j)) < dogs[idx].at<uchar>(Point(i + k, j + l)) &&
                            dogs[idx].at<uchar>(Point(i, j)) < dogs[idx-1].at<uchar>(Point(i + k, j + l)) &&
                            dogs[idx].at<uchar>(Point(i, j)) < dogs[idx+1].at<uchar>(Point(i + k, j + l)))
                            //extremaCount+=-1;
                            isminima = true;
                    else isnon = true;
                 } 
                }
                if(!(ismaxima ^ isminima) ||isnon) break;
                //cout<<"extremaCount : "<<extremaCount<<endl;

            }
            //if(std::abs(extremaCount)==8 && eliminateEdgeResp(i,j,dogs[idx]))  //Require one of them to be true only, If both are true the point is neither
            //cout<<extremaCount<<endl;           
            if((ismaxima ^ isminima) && !isnon && eliminateEdgeResp(i,j,dogs[idx]))  //Require one of them to be true only, If both are true the point is neither            
            {  //fill.at<uchar>(Point(i, j)) = 255; 
               KeyPoint K(i,j,pow(2,(idx+OctaveID)/s),-1,0,(int) OctaveID/s,idx); //KeyPoint(x,y,size,angle,response,octave,class_id)
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

/*
Scalar p = img.at<uchar>(Point(i, j));                   //Returns a vector of values 
Scalar(img.at<uchar>(Point(i, j))).val[0]; || p.val[0]  //Returnd a value of GrayScale pixel
fill.at<uchar>(Point(i, j)) = color;*/                 //Assign a color to point i,j
/*double min , max ; 
cv::minMaxLoc(dogs[2],&min,&max);
*/
//calculateOrientation
  /*int zero = 0;
  Mat gaussian = getGaussianKernel(64,1.5*size);
  for (int j = 0 ; j< gaussian.rows;j++)
  for(int i = 0 ; i < gaussian.cols ; i++)
  {   
      if(Scalar(gaussian.at<uchar>(Point(j,i))).val[0] != 0)
        {
          zero++;
        }


  }
  imwrite("./gaussiankernal.jpg",gaussian);
  cout<<"non zeros:  "<<zero<<"  newline"<<endl;
  //cout<<"Size of kernal: "<<gaussian.size<<"\trows: "<<gaussian.rows<<"\tcols: "<<gaussian.cols<<endl;
  */