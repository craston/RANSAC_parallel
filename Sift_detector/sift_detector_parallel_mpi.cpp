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
#include "SubplotImgs.hpp"
#include "omp.h"
#include "mpi.h"
#include <chrono>

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;
using namespace std::chrono;

void readme();
void sift_detect(Mat img,std::vector<KeyPoint> &kps);
void findExtrema(std::vector<Mat> dogs,std::vector<KeyPoint> &keypoints , double OctaveID);
void AssignOrientation(std::vector<KeyPoint> &keypoints, std::vector<Mat> blurredImgs);
void getBlurred(Mat img ,std::vector<Mat> &blurredImgs, double k);
void getDoGs(std::vector<Mat> blurredImgs ,std::vector<Mat> &dogs);
bool eliminateEdgeResp(int x , int y, Mat img);
int calculateOrientation( Mat img, Point2f k, double size);
int arg_max(int *x, int len);
void write_kps(Mat img,std::vector<KeyPoint> kps, int i);
void buff_to_kp(void *buff, std::vector<KeyPoint> &kp);
/*
 * @function main
 * @brief Main function
 */

#define s 6
#define ROW 3
#define COL 3
#define SIGMA 1.6
#define HIST_ROT_SIZE 36
int NB_OCTAVES;


typedef struct image_info {
        int rows;
        int cols;
        unsigned char *data;
} image;

typedef struct keypoint_info {
        int x;
        int y;
        float size;
        int angle;
        int nb_kp;
} kp_struct;

int main( int argc, char** argv )
{
/*-----------------------Init MPI---------------------------------------*/

  MPI_Init(&argc,&argv);
  int numTasks;
  int rank;
  MPI_Comm_size(MPI_COMM_WORLD, &numTasks);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  ///NUMBER OF OCTAVES IS LINEAR TO HOW MANY PROCESSES RUN
  NB_OCTAVES = numTasks;

/*-----------------------Read Image---------------------------------------*/

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

/*----------------Create Keypoints vector dataType to send between proc----------------*/

  //create vector cast
  MPI_Datatype types[5] = {MPI_INT, MPI_INT, MPI_FLOAT, MPI_INT, MPI_INT};
  MPI_Datatype mpi_kp_type;
  MPI_Datatype mpi_kp_vector; 
  MPI_Aint     offsets[5];
  int blocklengths[5] = {1,1,1,1,1};

  offsets[0] = offsetof(kp_struct, x);
  offsets[1] = offsetof(kp_struct, y);
  offsets[2] = offsetof(kp_struct, size);
  offsets[3] = offsetof(kp_struct, angle);
  offsets[4] = offsetof(kp_struct, nb_kp);

  //create keypoint type mpi
  MPI_Type_create_struct(4, blocklengths, offsets, types, &mpi_kp_type);
  MPI_Type_commit(&mpi_kp_type);
  //create vector type mpi
  MPI_Type_vector(2000,1,0,mpi_kp_type,&mpi_kp_vector);
  MPI_Type_commit(&mpi_kp_vector);

  std::vector<KeyPoint> keypoints; //store the keypoints of first octave
  Mat descriptors;
  high_resolution_clock::time_point t1 = high_resolution_clock::now();

/*-----------------------Start of Sift Process with MPI---------------------------------------*/


  float start_time = omp_get_wtime();
  if(rank > 0)
  {
    for (int i = 0; i< rank ; i++)
    {
      //int down_factor = pow(2,rank);
      //cout<<"DOWN FACTOR "<<down_factor<<" rank "<<rank<<endl;
      pyrDown(img, img, Size( img.cols/2, img.rows/2));
      cout<<"ROWS "<<img.rows<<"COLS "<<img.cols<<endl;
    }
  }

  sift_detect(img,keypoints);

  ///GATHER ALL KPS TO ONE KP
  kp_struct kps[keypoints.size()];
  for (int i = 0 ; i < keypoints.size() ; i++)
  {
    kps[i].x = keypoints[i].pt.x;
    kps[i].y = keypoints[i].pt.y;
    kps[i].size = keypoints[i].size;
    kps[i].angle = keypoints[i].angle;
    kps[i].nb_kp = keypoints.size();
  }

  kp_struct *rbuff;
  rbuff = (kp_struct *)malloc(2000*sizeof(kp_struct));
 
  cout<<"Keypoints detected, inside MPI realm: "<<keypoints.size()<<" from process "<<rank<<endl;
  if(rank!=0)
  {
    MPI_Send(kps,keypoints.size(),mpi_kp_vector,0,99,MPI_COMM_WORLD);
  }
  else
  {
    for(int i = 1 ; i< numTasks; i++)
    {
      MPI_Status status;
      MPI_Recv(rbuff,2000,mpi_kp_vector,i,99,MPI_COMM_WORLD,&status);
      cout<<"buff "<<rbuff[0].nb_kp<<" from process "<<i<<endl;
      buff_to_kp(rbuff, keypoints);
    }
  }
  
  ///timeit
  float run_time = omp_get_wtime() - start_time;
  high_resolution_clock::time_point t2 = high_resolution_clock::now();
  auto duration = duration_cast<microseconds>( t2 - t1 ).count();

  /*if(rank==0)
  {
    //Extract descriptors:
    Ptr<DescriptorExtractor> descriptorExtractor = SIFT::create();
    descriptorExtractor -> compute(img,keypoints,descriptors);

  }*/

  printf("RunTime walltime: %f \n",run_time);
  printf("RunTime chrono: %f \n",duration/1000000.0);
  

  MPI_Finalize();
  if(rank==0)
  cout<<"Final Keypoints detected: "<<keypoints.size()<<endl;
  if(rank==0)
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
void sift_detect(Mat img,std::vector<KeyPoint> &kps)
{
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  double i = 1 + s*rank;
  std::vector<Mat> blurredImgs;           //store blurred images of first octave    ------> s+3 =6 
  std::vector<Mat> dogs;            //store difference of gaussian images of first octave ----> s+2 = 5
  
  //initialize the octave with blurring the image
  getBlurred(img,blurredImgs,i);
  //apply difference of gaussian response and get the DoGs
  getDoGs(blurredImgs, dogs);
  //find the keypoints of the current octave
  findExtrema(dogs , kps,i);
  //Assign angles to keypoints
  AssignOrientation(kps,blurredImgs);
  //write_kps(img,kps,rank);
  blurredImgs.clear();
  dogs.clear();
  
  return;
}
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
  //std::vector<Mat> nmsDogs;
  bool ismaxima, isminima, isnon;
  int extremaCount = 0;
  int k, l, kbound, lbound, idx, i , j;  //filter indices

  //cout<<"dog size::" << dogs.size()<<endl;

#pragma omp  parallel for private(idx,i,j,ismaxima,isminima,isnon,kbound,lbound,k,l) shared(keypoints) 
  for (  idx = 1 ; idx < dogs.size()-1 ; idx++ )
  {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    cout<<"thread id: "<<omp_get_thread_num()<<" rank "<<rank<<endl;
    std::vector<KeyPoint> K_localCopy;
    for(  i = 0; i < dogsRows; i++ )
    {
        if (i == 0)
        { k = 0; kbound = 2; }
        else if (i == dogsRows-1)
        { k = -1; kbound = 1; }
        else
        { k = -1; kbound = 2; }

        for( j = 0; j < dogsCols; j++ )
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
            //cout<<extremaCount++<<"rows cols "<<i<<" "<<j<<" dogrows , dogcols "<<dogsRows<<" "<<dogsCols<<endl;
            //cout<<"entering loops"<<endl;
            for( ; k < kbound ; k++ )
            {
                for( ; l < lbound; l++ )
                {
                //Compare each dog image idx with image idx -1 and idx +1 and its neighbor
                  //cout<<"here1! "<<i+k<<" "<<j+l<<endl;
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
                //cout<<"pos1"<<endl;
                //cout<<"here2!"<<endl;
                if(!(ismaxima ^ isminima) || isnon) break;
                //scout<<"pos2"<<endl;
                //cout<<"extremaCount : "<<extremaCount<<endl;

            }

            //cout<<"exiting loops"<<endl;
            //if(std::abs(extremaCount)==8 && eliminateEdgeResp(i,j,dogs[idx]))  //Require one of them to be true only, If both are true the point is neither
            //cout<<extremaCount<<endl;    
            //#pragma omp critical
            //{      
            if((ismaxima ^ isminima) && !isnon && eliminateEdgeResp(i,j,dogs[idx]))  //Require one of them to be true only, If both are true the point is neither            
            {  //fill.at<uchar>(Point(i, j)) = 255; 

              KeyPoint K(i,j,pow(2,(idx+OctaveID)/s),-1,0,(int) OctaveID/s,idx); //KeyPoint(x,y,size,angle,response,octave,class_id)
              //cout<<"start!!"<<endl;
              K_localCopy.push_back(K);
              //cout<<"here!"<<endl;
            }
            //} 
       }
    }
    #pragma omp critical
    {
      keypoints.insert(keypoints.end(),K_localCopy.begin(),K_localCopy.end());
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

  dxx = img.at<uchar>(Point(x + 1, y)) + img.at<uchar>(Point(x - 1, y)) - 2*img.at<uchar>(Point(x, y));
  dyy = img.at<uchar>(Point(x, y + 1)) + img.at<uchar>(Point(x, y - 1)) - 2 * img.at<uchar>(Point(x, y));

  dxy = (img.at<uchar>(Point(x + 1, y + 1)) - img.at<uchar>(Point(x - 1, y + 1)) - img.at<uchar>(Point(x + 1,y - 1)) 
                    + img.at<uchar>(Point(x - 1, y - 1)))/ 2;
  TR  = dxx + dyy;
  DET = dxx*dyy - dxy*dxy;

  r = 10;
  th = (r+1)*(r+1)/r;
  ratioCurv = std::abs(TR*TR/DET);

  if(ratioCurv > th || DET < 0) return true;
  else               return false;

}



void AssignOrientation( std::vector<KeyPoint> &keypoints, std::vector<Mat> blurredImgs)
{ 
  double size;
  Point2f p;
  Mat img;
  int j ,i;
  float angle;
  std::vector<KeyPoint> newKeyPoints;

  #pragma omp parallelfor private(i,size,p,j,img,angle) shared(newKeyPoints)
  for(i = 0 ; i < keypoints.size() ; i++)
  {
    size = keypoints[i].size;
    //cout<<"size: "<<size<<endl;
    p     = keypoints[i].pt;          //keypoint location
    j     = keypoints[i].class_id;    //assigned blurred image index
    img   = blurredImgs[j];            
    angle = calculateOrientation(img,p,size);
    
    if(angle != -1)
    {
      keypoints[i].angle = angle;
      #pragma omp critical
      {
      newKeyPoints.push_back(keypoints[i]);
      }
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
        Lx = blurredCrop.at<uchar>(Point(i+1, j)) - blurredCrop.at<uchar>(Point(i-1, j));
        Ly = blurredCrop.at<uchar>(Point(i, j+1)) - blurredCrop.at<uchar>(Point(i, j-1));
        //Lx = cropped.at<uchar>(Point(i+1, j)) - cropped.at<uchar>(Point(i-1, j));
        //Ly = cropped.at<uchar>(Point(i, j+1)) - cropped.at<uchar>(Point(i, j-1));
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

void buff_to_kp(void *rbuff, std::vector<KeyPoint> &kp)
{
  kp_struct *buff = (kp_struct *)rbuff;
  for(int i=0 ; i< buff[0].nb_kp ; i++)
  {
    int x = buff[i].x , y = buff[i].y , angle = buff[i].angle;
    float size = buff[i].size;
    KeyPoint K(x,y,size,angle,0,0,-1);
    kp.push_back(K);
  }
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

/*MPI_IMAGE_TYPE
  MPI_Datatype types[3] = {MPI_INT, MPI_INT, MPI_UNSIGNED_CHAR};
  MPI_Datatype mpi_img_type;
  MPI_Aint     offsets[3];
  int blocklengths[3] = {1,1,img.rows*img.cols};

  offsets[0] = offsetof(image, rows);
  offsets[1] = offsetof(image, cols);
  offsets[2] = offsetof(image, data);

  MPI_Type_create_struct(3, blocklengths, offsets, types, &mpi_img_type);
  MPI_Type_commit(&mpi_img_type);

  for (int i = 0 ; i < numTasks ; i++)
  {
    if (rank == 0)
    {
      image bcast_image;
      bcast_image.rows = img.rows;
      bcast_image.cols = img.cols;
      bcast_image.data = img.data;

      MPI_Bcast(&bcast_image,3, mpi_img_type,0, MPI_COMM_WORLD);
    }
    else
    {
      MPI_Status status;
      int src = 0;
      image recv;

      MPI_Recv(&recv,   1, mpi_img_type, src, -1, MPI_COMM_WORLD, &status);
      printf("Rank %d: Received: cols = %d rows = %d\n", rank, recv.rows,
               recv.cols);
    }
  }
  */

 /*
  int displs[numTasks];
  int recvcounts[numTasks];
  for (int i = 0 ; i < numTasks ; i++)
  {
    displs[i] = i*2000;
    recvcounts[i] = 2000;
  }
  //MPI_Gatherv(kps,keypoints.size(), mpi_kp_vector, rbuf, recvcounts, displs, mpi_kp_vector, 0, MPI_COMM_WORLD); 
  //MPI_Gather(kps,keypoints.size(), mpi_kp_vector, rbuf, 2000, mpi_kp_vector, 0, MPI_COMM_WORLD); 
  */