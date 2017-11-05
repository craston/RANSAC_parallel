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

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;
void readme();
bool contains(vector<int> rand_idx, int num);
float distance(float x1, float y1, float x2, float y2);
/*
 * @function main
 * @brief Main function
 */

// Creating struture for 3d point
#define ROW 3
#define COL 3

struct pt3D{
  float x,y,z;
};

int main( int argc, char** argv )
{
  if( argc != 3 )
  { 
    readme();
    return -1; 
  }

  Mat img_1 = imread( argv[1], IMREAD_GRAYSCALE );
  Mat img_2 = imread( argv[2], IMREAD_GRAYSCALE );
  
  if( !img_1.data || !img_2.data )
  { 
    std::cout<< " --(!) Error reading images " << std::endl; return -1; 
  }
  
  //-- Step 1: Detect the keypoints using SURF Detector, compute the descriptors
  int minHessian = 400;
  Ptr<SURF> detector = SURF::create();
  detector->setHessianThreshold(minHessian);
  std::vector<KeyPoint> keypoints_1, keypoints_2;
  Mat descriptors_1, descriptors_2;
  detector->detectAndCompute( img_1, Mat(), keypoints_1, descriptors_1 );
  detector->detectAndCompute( img_2, Mat(), keypoints_2, descriptors_2 );
  
  //-- Step 2: Matching descriptor vectors using FLANN matcher
  FlannBasedMatcher matcher;
  std::vector< DMatch > matches;
  matcher.match( descriptors_1, descriptors_2, matches );
  double max_dist = 0; double min_dist = 100;
  //-- Quick calculation of max and min distances between keypoints
  
  for( int i = 0; i < descriptors_1.rows; i++ )
  { 
    double dist = matches[i].distance;
    if( dist < min_dist ) min_dist = dist;
    if( dist > max_dist ) max_dist = dist;
  }
  printf("-- Max dist : %f \n", max_dist );
  printf("-- Min dist : %f \n", min_dist );
  

  //-- Draw only "good" matches (i.e. whose distance is less than 2*min_dist,
  //-- or a small arbitary value ( 0.02 ) in the event that min_dist is very
  //-- small)
  //-- PS.- radiusMatch can also be used here.
  std::vector< DMatch > good_matches;
  for( int i = 0; i < descriptors_1.rows; i++ )
  { 
    if( matches[i].distance <= max(2*min_dist, 0.02) )
    { 
      good_matches.push_back( matches[i]); 
    }
  }
  
  //-- Draw only "good" matches
  Mat img_matches;
  drawMatches( img_1, keypoints_1, img_2, keypoints_2,
               good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
               vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
  //-- Show detected matches
  imshow( "Good Matches", img_matches );
  
  for( int i = 0; i < (int)good_matches.size(); i++ )
  { 
    printf( "-- Good Match [%d] Keypoint 1: %d  -- Keypoint 2: %d  \n", i, good_matches[i].queryIdx, good_matches[i].trainIdx ); 
  }

  // --- RANSAC Algorithm Serial
  std::vector<Point3f> src_pts;
  std::vector<Point3f> dst_pts;
  Point3f temp;
  int N = (int)good_matches.size();

  for( int i = 0; i < N; i++ ){
    temp.x = keypoints_1[good_matches[i].queryIdx].pt.x;
    temp.y = keypoints_1[good_matches[i].queryIdx].pt.y;
    temp.z = 1;
    src_pts.push_back(temp);

    temp.x = keypoints_2[good_matches[i].trainIdx].pt.x;
    temp.y = keypoints_2[good_matches[i].trainIdx].pt.y;
    dst_pts.push_back(temp);
    printf("Source points = [%f,%f,%f]\n", src_pts[i].x, src_pts[i].y, src_pts[i].z);
  }

  std::vector<Point3f> src_random, dst_random;
  int outliers, num;
  int past_outliers = N +1;
  std::vector<int> rand_idx_src, rand_idx_dst, idx_remove, Best_idx_remove;
  std::vector< DMatch > best_matches = good_matches;
  std::vector< KeyPoint > best_keypoints_1 = keypoints_1;
  std::vector< KeyPoint > best_keypoints_2 = keypoints_2;
  Mat H, Best_H;

  for(int k =0; k<2000; k++){
    // Generating 3 random src_points
    cout<<"ITERATION ="<<k<<endl;
    idx_remove.clear();
    rand_idx_src.clear();
    rand_idx_dst.clear();
    outliers = 0;
    
    for(int j = 0; j<3; j++){
      do{num = rand()%N;}while (contains(rand_idx_src, num));
      
      rand_idx_src.push_back(num);
      src_random.push_back(src_pts[num]);
      cout<<"source random index ["<<j<<"]:"<< num << endl;

      do{num = rand()%N;}while (contains(rand_idx_dst, num));
      rand_idx_dst.push_back(num);;
      dst_random.push_back(dst_pts[num]);
      cout<<"destination random index ["<<j<<"]:"<< num << endl;
    }
    H = findHomography(src_random, dst_random);
    cout<<"H = "<<H<<endl;

    for(int i = 0 ; i<N ; i++){
      Point2f temp;
      temp.x = H.at<double>(0,0)*src_pts[i].x + H.at<double>(0,1)*src_pts[i].y + H.at<double>(0,2)*src_pts[i].z;
      temp.y = H.at<double>(1,0)*src_pts[i].x + H.at<double>(1,1)*src_pts[i].y + H.at<double>(1,2)*src_pts[i].z;
      //cout<<"H "<<H.at<double>(1,0)<<" "<<H.at<double>(1,1)<<" "<<H.at<double>(1,1)<<endl;
      //cout<<"source "<<src_pts[i].x<<" "<<src_pts[i].y<<" "<<src_pts[i].z<<endl;
      //cout<<"temp.x "<<temp.x<<" temp.y "<<temp.y<<endl;

      //cout<<"distance = "<<distance(temp.x, temp.y, dst_pts[i].x, dst_pts[i].y )<<endl;
      if(distance(temp.x, temp.y, dst_pts[i].x, dst_pts[i].y )> 300 ){
        outliers += 1;
        idx_remove.push_back(i);
      }
    }
    cout<<"Number"<< outliers<<endl;
    if(outliers < past_outliers){
      cout<<"iteration["<<k<<"]: outliers = "<<outliers;
      past_outliers = outliers;
      Best_H = H;
      Best_idx_remove = idx_remove;
    }
  } 

  for(int i =0; i< (int)Best_idx_remove.size(); i++){
      cout<<"ITERATION = "<<i<<endl;
      best_matches.erase(best_matches.begin() + Best_idx_remove[i] - i);
  }
  drawMatches( img_1, best_keypoints_1, img_2, best_keypoints_2,
               best_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
               vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
  //-- Show detected matches
  imshow( "Best Matches", img_matches );

  waitKey(0);
  return 0;
}
/*
 * @function readme
 */
void readme()
{ 
  std::cout << " Usage: ./SURF_FlannMatcher <img1> <img2>" << std::endl; 
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
