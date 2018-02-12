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
#include "omp.h"
#include <random>
#define nb_experiments 5

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

struct time_best
{
    double time;
    std::vector< DMatch > best_matches;
};

void readme();
bool contains(vector<int> rand_idx, int num);
float distance(float x1, float y1, float x2, float y2);
time_best RANSAC_parallel(Mat img_1, Mat img_2, std::vector<Point3f> src_vec, 
	std::vector<Point3f> dst_vec, std::vector<Point3f> src_pts, std::vector<Point3f> dst_pts, std::vector<DMatch> good_matches, int ITER);

int main( int argc, char** argv ){
	if( argc != 5 ){ 
		readme();
		return -1; 
	}

	Mat img_1 = imread( argv[1], IMREAD_GRAYSCALE );
	Mat img_2 = imread( argv[2], IMREAD_GRAYSCALE );

	if( !img_1.data || !img_2.data ){ 
		std::cout<< " --(!) Error reading images " << std::endl; return -1; 
	}

	//-- Step 1: Detect the keypoints using SIFT Detector, compute the descriptors
	Ptr<SIFT> detector = SIFT::create();
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
	for( int i = 0; i < descriptors_1.rows; i++ ){ 
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
	for( int i = 0; i < descriptors_1.rows; i++ ){ 
		if( matches[i].distance <= max(atoi(argv[4])*min_dist, 0.02) ){ 
	  		good_matches.push_back( matches[i]); 
		}
	}

	//-- Draw only "good" matches
	Mat img_matches;
	drawMatches( img_1, keypoints_1, img_2, keypoints_2, good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
	           vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
	//-- Show detected matches
	imshow( "Good Matches", img_matches );

	/*for( int i = 0; i < (int)good_matches.size(); i++ )
	{ 
	//printf( "-- Good Match [%d] Keypoint 1: %d  -- Keypoint 2: %d  \n", i, good_matches[i].queryIdx, good_matches[i].trainIdx ); 
	}*/
	cout<<"Number of  matches : "<<(int)matches.size()<<endl;
	cout<<"Number of Good matches : "<<(int)good_matches.size()<<endl;

	std::vector<Point3f> src_pts, dst_pts;
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
		//printf("Source points = [%f,%f,%f]\n", src_pts[i].x, src_pts[i].y, src_pts[i].z);
	}

	std::vector<Point3f> src_vec, dst_vec;
	std::vector<int> rand_idx_src, rand_idx_dst;
	int num;
	// srand(1);

	for(int k = 0; k< atoi(argv[3]); k++){
		rand_idx_src.clear();
		rand_idx_dst.clear();
		for(int j = 0; j<4; j++){
			do{

				num = rand()%N;
				/*
				std::random_device rd;
			  	std::mt19937 mt(rd());
			    std::uniform_int_distribution<int> distribution(0,N-1);    
			    num = distribution(rd);
			    */
			}while (contains(rand_idx_src, num));

			rand_idx_src.push_back(num);
			src_vec.push_back(src_pts[num]);
			//cout<<"source random index ["<<j<<"]:"<< num << endl;

			do{
				num = rand()%N;
				/*
				std::random_device rd;
			  	std::mt19937 mt(rd());
			    std::uniform_int_distribution<int> distribution(0,N-1);    
			    num = distribution(rd);
			    */
			}while (contains(rand_idx_dst, num));
			
			rand_idx_dst.push_back(num);
			dst_vec.push_back(dst_pts[num]);
			//cout<<"destination random index ["<<j<<"]:"<< num << endl;
	  	}
	}

	double time = 0;
	time_best result;
	//RUNNING EXPERIMENTS
	for(int i = 0; i<nb_experiments; i++){
		result = RANSAC_parallel(img_1, img_2, src_vec, dst_vec, src_pts, dst_pts, good_matches, atoi(argv[3]));
		time += result.time;
	}
	printf("Average time = %f\n", time/nb_experiments);

	std::vector<DMatch> best_matches = result.best_matches;

	//Checking for serial and parallel implementation
	for(int i = 0; i < 5; i++){
		cout<<keypoints_1[best_matches[i].queryIdx].pt.x<<endl;
	}

	drawMatches( img_1, keypoints_1, img_2, keypoints_2, best_matches, img_matches, Scalar::all(-1), Scalar::all(-1),vector<char>(), 
		DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
	//-- Show detected best matches
	//imshow( "Best Matches", img_matches );
	imwrite("Result.jpg", img_matches);
	//waitKey(0);

	return 0;

}

time_best RANSAC_parallel(Mat img_1, Mat img_2, std::vector<Point3f> src_vec, std::vector<Point3f> dst_vec, 
	std::vector<Point3f> src_pts, std::vector<Point3f> dst_pts, std::vector<DMatch> good_matches, int ITER){

  	// --- RANSAC Algorithm Serial
	int N = (int)good_matches.size();
	std::vector<Point3f> src_random, dst_random;
	int outliers;
	int past_outliers = N +1;
	std::vector<int> idx_remove, Best_idx_remove;
	std::vector< DMatch > best_matches = good_matches;
	Mat H;

	register unsigned int i,j,k;
	double start_time = omp_get_wtime();

	#pragma omp parallel for schedule(dynamic) private(i, j, k, idx_remove, H, src_random, dst_random, outliers)

	for(k =0; k<ITER; k++){
		idx_remove.clear();
		src_random.clear();
		dst_random.clear();
		outliers = 0;

		//src_vec and dst_vec are shared between threads
		#pragma omp critical
		{
			for(j = 0; j<4; j++){
			  src_random.push_back(src_vec.back());
			  src_vec.pop_back();
			  dst_random.push_back(dst_vec.back());
			  dst_vec.pop_back();
			}
		}
		//cout<<"Thread number "<< omp_get_thread_num()<<" " <<rand_idx_src[0]<<", "<<rand_idx_src[1]<<", "<<rand_idx_src[2]<<endl<<flush;

		H = findHomography(src_random, dst_random);
		
		for(i = 0 ; i<N ; i++){
			Point2f temp2;
			temp2.x = H.at<double>(0,0)*src_pts[i].x + H.at<double>(0,1)*src_pts[i].y + H.at<double>(0,2)*src_pts[i].z;
			temp2.y = H.at<double>(1,0)*src_pts[i].x + H.at<double>(1,1)*src_pts[i].y + H.at<double>(1,2)*src_pts[i].z;

			//cout<<distance(temp2.x, temp2.y, dst_pts[i].x, dst_pts[i].y )<<endl;
			if(distance(temp2.x, temp2.y, dst_pts[i].x, dst_pts[i].y )> 100){
				outliers += 1;
				idx_remove.push_back(i);
			}
		}
		//cout<<"Number of outliers: "<< outliers<<endl<<flush;
		#pragma omp critical
		{
		  	if(outliers < past_outliers){
			    cout<<"Thread number "<< omp_get_thread_num()<<" iteration["<<k<<"]: "<<outliers<<endl<<flush;
			    past_outliers = outliers;
			    Best_idx_remove = idx_remove;
		  	}
		}
	} 

	double run_time = omp_get_wtime() - start_time;
	for(int i =0; i< (int)Best_idx_remove.size(); i++){
	  	best_matches.erase(best_matches.begin() + Best_idx_remove[i] - i);
	}

	time_best result;
	result.time = run_time;
	result.best_matches = best_matches;

	return result;
}
/*
 * @function readme
 */
void readme()
{ 
  std::cout << " Usage: ./SIFT_FlannMatcher <img1> <img2>" << std::endl; 
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
