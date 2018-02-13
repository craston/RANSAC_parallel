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
#include "omp.h"
#include <random>
#include "mpi.h"
// #include <iterator>
// #include <climits>

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

void readme();
void sift_detect_full(Mat img,std::vector<KeyPoint> &kps);
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
void match( Mat img1 , Mat img2, std::vector<KeyPoint> keypoints_1, std::vector<KeyPoint> keypoints_2, Mat descriptors_1 , Mat descriptors_2);
bool contains(vector<int> rand_idx, int num);
float distance(float x1, float y1, float x2, float y2);


#define s 6
#define ROW 3
#define COL 3
#define SIGMA 1.6
#define HIST_ROT_SIZE 36
#define ITER 1000
int NB_OCTAVES;


struct DMatch_new{
  int queryIdx;
  int trainIdx;
  float distance;
};

struct time_best
{
    double time;
    int outliers;
    std::vector< DMatch_new > best_matches;
};

time_best RANSAC_parallel(std::vector<Point3f> src_vec, std::vector<Point3f> dst_vec,std::vector<Point3f> src_pts, std::vector<Point3f> dst_pts, std::vector<DMatch_new> good_matches, int chunk_size);

typedef struct keypoint_info {
        float x;
        float y;
        float size;
        float angle;
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

  if( argc != 3 )
  { 
    readme();
    return -1; 
  }

  Mat img1 = imread( argv[1], IMREAD_GRAYSCALE );

  Mat img2 = imread( argv[2], IMREAD_GRAYSCALE );

  
  if( !img1.data || !img2.data)
  { 
    std::cout<< " --(!) Error reading image " << std::endl; return -1; 
  }

/*----------------Create Keypoints vector dataType to send between proc----------------*/

  //create vector cast
  float start_time = omp_get_wtime();
  std::vector<KeyPoint> keypoints1 , keypoints2;
  Mat descriptors1 ,descriptors2;
  sift_detect_full(img1,keypoints1);
  sift_detect_full(img2,keypoints2);

  MPI_Barrier(MPI_COMM_WORLD);
  cout<<"Keypoint detection done-------Commence Matching"<<endl;
  if(rank==0)
  {
    //Extract descriptors:
    Ptr<DescriptorExtractor> descriptorExtractor = SIFT::create();
    descriptorExtractor -> compute(img1,keypoints1,descriptors1);
    descriptorExtractor -> compute(img2,keypoints2,descriptors2);
  }
  
  if(rank==0)
  {
    cout<<"Final Keypoints detected: "<<keypoints1.size()<<endl;
    cout<<"DescriptorExtractor created with size: "<<descriptors1.size()<<endl;
    cout<<"Final Keypoints detected: "<<keypoints2.size()<<endl;
    cout<<"DescriptorExtractor created with size: "<<descriptors2.size()<<endl;
  }
  /*------match----*/
  // std::vector< DMatch > good_matches;
  MPI_Barrier(MPI_COMM_WORLD);
  match(img1,img2,keypoints1,keypoints2,descriptors1,descriptors2);
  float run_time = omp_get_wtime() - start_time;
  cout<<"Total Runtime for SIFT and RANSAC = "<<run_time<<endl<<std::flush;
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();

  // waitKey(0);
  return 0;
}

/*
 * @function readme
 */

void readme()
{ 
  std::cout << " Usage: ./sift_Ransac <img1>  <img2>IMG missing " << std::endl; 
}
/*
 * @functions SIFT
 */
void sift_detect_full(Mat img, std::vector<KeyPoint> &keypoints)
{
  int numTasks;
  int rank;
  MPI_Comm_size(MPI_COMM_WORLD, &numTasks);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Datatype types[5] = {MPI_FLOAT, MPI_FLOAT, MPI_FLOAT, MPI_FLOAT, MPI_INT};
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

  //std::vector<KeyPoint> keypoints; //store the keypoints of first octave

/*-----------------------Start of Sift Process with MPI---------------------------------------*/


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

  //printf("RunTime walltime: %f \n",run_time);
  
}
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
  std::vector<KeyPoint> local_Kp,global_Kp;

  #pragma omp parallel shared(global_Kp) private(local_Kp)
  {
    #pragma omp for private(i,size,p,j,img,angle)
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
        local_Kp.push_back(keypoints[i]);
        //cout<<"thread "<<omp_get_thread_num()<<" assigned angle "<<angle<<" to keypoint "<< i << endl;
      }
    }
    #pragma omp critical
    { 
      cout<<"local size "<<local_Kp.size()<<endl;
      global_Kp.insert(global_Kp.end(),local_Kp.begin(),local_Kp.end());
      cout<<"Thead "<<omp_get_thread_num()<<" global assigned "<<global_Kp.size()<<endl;
    }
  }
  keypoints.clear();

  keypoints = global_Kp;
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
/*--------------------------------------------RANSAC MATCHING MPART-------------------------*/


void match( Mat img1 , Mat img2, std::vector<KeyPoint> keypoints_1, std::vector<KeyPoint> keypoints_2, Mat descriptors_1 , Mat descriptors_2){

  MPI_Status status;
  
  //Define a new MPI datatype for struct Point3f
  int num_members = 3;          // Number of members in struct
  int lengths[num_members] = { 1, 1, 1};      // Length of each member in struct
  MPI_Aint offsets[num_members] = {0, sizeof(float), 2*sizeof(float)};
  MPI_Datatype types[num_members] = { MPI_FLOAT, MPI_FLOAT, MPI_FLOAT};
  MPI_Datatype Point3f_type;
  MPI_Type_struct(num_members, lengths, offsets, types, &Point3f_type);
  MPI_Type_commit(&Point3f_type);
  
  double time = 0;
  time_best result;
  //Define a new MPI datatype for struct DMatch
  int num_membersD = 3;                 // Number of members in struct
  int lengthsD[num_membersD] = { 1, 1, 1};      // Length of each member in struct
  MPI_Aint offsetsD[num_membersD] = {0, sizeof(int), 2*sizeof(int)};
  MPI_Datatype typesD[num_membersD] = { MPI_INT, MPI_INT, MPI_FLOAT};
  MPI_Datatype DMatch_type;
  MPI_Type_struct(num_membersD, lengthsD, offsetsD, typesD, &DMatch_type);
  MPI_Type_commit(&DMatch_type);
  int my_rank, world_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  int chunk_size = 4*ITER/(world_size - 1);
  std::vector< DMatch > matches, good_matches;
  int outliers_collect[world_size];
  int N;
 
  if(my_rank == 0){
    FlannBasedMatcher matcher;
    // std::vector< DMatch > matches;
    matcher.match( descriptors_1, descriptors_2, matches );
    double max_dist = 0; double min_dist = 100;
    
    //-- Quick calculation of max and min distances between keypoints
    for( int i = 0; i < descriptors_1.rows; i++ ){ 
      double dist = matches[i].distance;
      if( dist < min_dist ) min_dist = dist;
      if( dist > max_dist ) max_dist = dist;
    }
    printf("-- Max dist : %f \n", max_dist );
    printf("-- Min dist : %f \n", min_dist );

    for( int i = 0; i < descriptors_1.rows; i++ ){ 
      if( 0.0< matches[i].distance <= max(100.0, 0.02) ){ 
          good_matches.push_back( matches[i]); 
      }
    }

    cout<<"Number of  matches : "<<(int)matches.size()<<endl;
    cout<<"Number of Good matches : "<<(int)good_matches.size()<<endl;

    std::vector<Point3f> src_pts, dst_pts;
    Point3f temp;
    N = (int)good_matches.size();

    for( int i = 0; i < N; i++ ){
      temp.x = keypoints_1[good_matches[i].queryIdx].pt.x;
      temp.y = keypoints_1[good_matches[i].queryIdx].pt.y;
      temp.z = 1;
      src_pts.push_back(temp);

      temp.x = keypoints_2[good_matches[i].trainIdx].pt.x;
      temp.y = keypoints_2[good_matches[i].trainIdx].pt.y;
      temp.z = 1;
      dst_pts.push_back(temp);
    }

    std::vector<Point3f> src_vec, dst_vec;
    std::vector<int> rand_idx_src, rand_idx_dst;
    int num;
    srand(1);

    for(int k = 0; k< ITER ; k++){
      rand_idx_src.clear();
      rand_idx_dst.clear();
      for(int j = 0; j<4; j++){
          do{
              num = rand()%N;
          }while (contains(rand_idx_src, num));

          rand_idx_src.push_back(num);
          src_vec.push_back(src_pts[num]);
        
          do{
              num = rand()%N;
          }while (contains(rand_idx_dst, num));

          rand_idx_dst.push_back(num);
          dst_vec.push_back(dst_pts[num]);
      }
    }

    //Process 0 will now divide the src and dst vector and points of good matches between all the process
    //Converting vectors to arrays before broadcasting
    Point3f *src_vec_a = &src_vec[0];
    Point3f *dst_vec_a = &dst_vec[0];
    Point3f *src_pts_a = &src_pts[0];
    Point3f *dst_pts_a = &dst_pts[0];
    int i,j;
    
    DMatch_new good_matches_struct[N];
    for( int i = 0; i < N; i++){
      good_matches_struct[i].queryIdx = good_matches[i].queryIdx;
      good_matches_struct[i].trainIdx = good_matches[i].trainIdx;
      good_matches_struct[i].distance = good_matches[i].distance;         
    }
    
    cout<<"my_rank = "<<my_rank<<" src_vec = "<<src_vec_a[0].x<<endl<<std::flush;

    for(i = 0; i<world_size-1; i++){
      MPI_Send(&src_vec_a[i*chunk_size], chunk_size, Point3f_type, i+1, 0, MPI_COMM_WORLD);
      MPI_Send(&dst_vec_a[i*chunk_size], chunk_size, Point3f_type, i+1, 1, MPI_COMM_WORLD);
      
      MPI_Send(&N, 1, MPI_INT, i+1, 99, MPI_COMM_WORLD);

      MPI_Send(&src_pts_a[0], N, Point3f_type, i+1, 2, MPI_COMM_WORLD);
      MPI_Send(&dst_pts_a[0], N, Point3f_type, i+1, 3, MPI_COMM_WORLD);
      
      MPI_Send(&good_matches_struct[0], N, DMatch_type, i+1, 100, MPI_COMM_WORLD);
    }
  }
  else{
    time = 0;
    Point3f src_vec_a[chunk_size], dst_vec_a[chunk_size];
    MPI_Recv(&(src_vec_a[0]), chunk_size, Point3f_type, 0, 0, MPI_COMM_WORLD, &status);
    MPI_Recv(&(dst_vec_a[0]), chunk_size, Point3f_type, 0, 1, MPI_COMM_WORLD, &status);
    cout<<"my_rank = "<<my_rank<<" src_vec = "<<src_vec_a[0].x<<endl<<std::flush;

    int N;
    MPI_Recv(&N, 1, MPI_INT, 0, 99, MPI_COMM_WORLD, &status);
    cout<<"my_rank = "<<my_rank<<" sizeBcast = "<<N<<endl<<std::flush;

    Point3f src_pts_a[N], dst_pts_a[N];
    MPI_Recv(&(src_pts_a[0]), N, Point3f_type, 0, 2, MPI_COMM_WORLD, &status);
    MPI_Recv(&(dst_pts_a[0]), N, Point3f_type, 0, 3, MPI_COMM_WORLD, &status);
    cout<<"my_rank = "<<my_rank<<" src_pts = "<<src_pts_a[0].x<<endl<<std::flush;

    DMatch_new good_matches_struct[N];
    MPI_Recv(&(good_matches_struct[0]), N, DMatch_type, 0, 100, MPI_COMM_WORLD, &status);
    cout<<"my_rank = "<<my_rank<<" good_matches = "<<good_matches_struct[0].queryIdx<<endl<<std::flush;

    std::vector<Point3f> src_vec_new(src_vec_a, src_vec_a + sizeof(src_vec_a)/sizeof(src_vec_a[0]));
    std::vector<Point3f> dst_vec_new(dst_vec_a, dst_vec_a + sizeof(dst_vec_a)/sizeof(dst_vec_a[0]));
    std::vector<Point3f> src_pts_new(src_pts_a, src_pts_a + sizeof(src_pts_a)/sizeof(src_pts_a[0]));
    std::vector<Point3f> dst_pts_new(dst_pts_a, dst_pts_a + sizeof(dst_pts_a)/sizeof(dst_pts_a[0]));
    std::vector<DMatch_new> good_matches_new(good_matches_struct, good_matches_struct + sizeof(good_matches_struct)/sizeof(good_matches_struct[0]));
    printf("MY rank %d\n", my_rank);

    result = RANSAC_parallel(src_vec_new, dst_vec_new, src_pts_new, dst_pts_new, good_matches_new, chunk_size);
    printf("Rank : %d, time = %f, outliers = %d\n", my_rank, result.time, result.outliers);
  }
  MPI_Gather(&result.outliers, 1, MPI_INT, outliers_collect, 1, MPI_INT, 0, MPI_COMM_WORLD); 
  int min;
  int proc_idx;

  if(my_rank == 0 ){
    min = INT_MAX;
    for(int i = 1; i< world_size; i ++){
      printf("List of outlier = %d \n",outliers_collect[i]);
      if (outliers_collect[i]<min){
        min = outliers_collect[i];
        proc_idx = i;
      }
    }
    printf("mininum = %d , %d\n", proc_idx, min );
  }

  MPI_Bcast(&proc_idx, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
  
  if(my_rank == proc_idx){
    DMatch_new *Final_matches = &(result.best_matches[0]);
    cout<<N<<" "<<my_rank<<endl<<std::flush;

    MPI_Send(&Final_matches[0], N, DMatch_type, 0, 200, MPI_COMM_WORLD);
  }

  if(my_rank == 0){
    cout<<N<<" "<<my_rank<<endl<<std::flush;
    DMatch_new Final_matches[N];
    MPI_Recv(&Final_matches[0], N, DMatch_type, proc_idx, 200, MPI_COMM_WORLD, &status);
    //Checking for serial and parallel implementation

    DMatch match_to_display[N-min];
    for(int i = 0; i < N-min; i++){
        match_to_display[i].queryIdx = 0;
        match_to_display[i].trainIdx = 0 ;
        match_to_display[i].distance = 0.0;
    }
    int j = 0;
    for(int i = 0; i < N; i++){
      if(Final_matches[i].queryIdx !=0 && Final_matches[i].trainIdx != 0 && Final_matches[i].distance != 0){
        match_to_display[j].queryIdx = Final_matches[i].queryIdx ;
        match_to_display[j].trainIdx = Final_matches[i].trainIdx ;
        match_to_display[j].distance = Final_matches[i].distance ;
        j++;
      }

    }
    cout<<"total Best "<<j<<endl<<std::flush;

    std::vector<DMatch> display_points(match_to_display, match_to_display + sizeof(match_to_display)/sizeof(match_to_display[0]));
    cout<<"Length of final matches = "<<N-min<<endl<<std::flush;
    Mat img_matches;
    
    cout<<"matches size "<<matches.size()<<endl<<std::flush;
    drawMatches( img1, keypoints_1, img2, keypoints_2, display_points, img_matches, Scalar::all(-1), Scalar::all(-1),vector<char>(), 
     DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
 
    imwrite("Result_sift_RANSAC.jpg", img_matches);
  }
}

time_best RANSAC_parallel(std::vector<Point3f> src_vec, std::vector<Point3f> dst_vec, 
    std::vector<Point3f> src_pts, std::vector<Point3f> dst_pts, std::vector<DMatch_new> good_matches, int chunk_size){

    // --- RANSAC Algorithm Serial
    int N = (int)good_matches.size();
    cout<<"good_matches size = "<<N<<endl<<std::flush;
    cout<<"src_vec size = "<<(int)src_vec.size()<<endl<<std::flush;
    cout<<"src_pts size = "<<(int)src_pts.size()<<endl<<std::flush;
    cout<<"chunk_size = "<<chunk_size<<endl<<std::flush;
    std::vector<Point3f> src_random, dst_random;
    int outliers;
    int past_outliers = N +1;
    std::vector<int> idx_remove, Best_idx_remove;
    std::vector< DMatch_new > best_matches = good_matches;
    Mat H;

    register unsigned int i,j,k;
    double start_time = omp_get_wtime();

    #pragma omp parallel for schedule(static) private(i, j, k, idx_remove, H, src_random, dst_random, outliers)
    for(k =0; k < chunk_size/4; k++){
        // cout<<k<<endl<<std::flush;
        idx_remove.clear();
        src_random.clear();
        dst_random.clear();
        outliers = 0;

        #pragma omp critical
        {
            for(j = 0; j<4; j++){
                // cout<<"hi"<<endl<<std::flush;
                src_random.push_back(src_vec.back());
                src_vec.pop_back();
                dst_random.push_back(dst_vec.back());
                dst_vec.pop_back();
            }
        }
        // cout<<"h1i"<<endl<<std::flush;
        H = findHomography(src_random, dst_random);
        // cout<<"H: :"<<H.size()<<" hi2"<<endl<<std::flush;
        for(i = 0 ; i < N ; i++){
            Point2f temp2;
            temp2.x = H.at<double>(0,0)*src_pts[i].x + H.at<double>(0,1)*src_pts[i].y + H.at<double>(0,2)*src_pts[i].z;
            temp2.y = H.at<double>(1,0)*src_pts[i].x + H.at<double>(1,1)*src_pts[i].y + H.at<double>(1,2)*src_pts[i].z;

            // cout<<distance(temp2.x, temp2.y, dst_pts[i].x, dst_pts[i].y )<<endl<<std::flush;
            if(distance(temp2.x, temp2.y, dst_pts[i].x, dst_pts[i].y )> 100){
                outliers += 1;
                idx_remove.push_back(i);
            }
        }
        // cout<<"Number of outliers: "<< outliers<<endl<<flush;
        #pragma omp critical
        { 
          // cout<<k<<endl;
            if(outliers < past_outliers){
                cout<<" Thread number "<< omp_get_thread_num()<<" iteration["<<k<<"]: "<<outliers<<endl<<flush;
                past_outliers = outliers; 
                Best_idx_remove = idx_remove;
            }
        }
    } 

    double run_time = omp_get_wtime() - start_time;
    for(int i =0; i< (int)Best_idx_remove.size(); i++){
         best_matches[Best_idx_remove[i]].queryIdx = 0;
         best_matches[Best_idx_remove[i]].trainIdx = 0;
         best_matches[Best_idx_remove[i]].distance = 0.0;
    }

    time_best result;
    result.time = run_time;
    result.outliers = past_outliers;
    result.best_matches = best_matches;

    return result;
}
/*
 * @function contains
 */
bool contains(std::vector<int> rand_idx, int num){
    return std::find(rand_idx.begin(), rand_idx.end(), num) != rand_idx.end();
}

float distance(float x1, float y1, float x2, float y2){
    return(sqrt(pow(x1 - x2, 2) + pow(y1 - y2,2)));
}
