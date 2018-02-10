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
#include "mpi.h"
#include <iterator>

#define nb_experiments 1

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

struct DMatch_new{
	int queryIdx;
	int trainIdx;
	float distance;
};
/*Struct of Dmatch

// struct Point3f
// {
// 	float x;
// 	float y;
// 	float z;
// };	
*/
struct time_best
{
    double time;
    std::vector< DMatch_new > best_matches;
};

void readme();
bool contains(vector<int> rand_idx, int num);
float distance(float x1, float y1, float x2, float y2);
time_best RANSAC_parallel(std::vector<Point3f> src_vec, std::vector<Point3f> dst_vec, 
    std::vector<Point3f> src_pts, std::vector<Point3f> dst_pts, std::vector<DMatch_new> good_matches, int chunk_size);

int main( int argc, char** argv ){
    MPI_Init(NULL, NULL);

    // Get the number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    cout<<"world_size = "<<world_size<<"\n"<<std::flush;
    if(1000%(world_size - 1) != 0 || (world_size - 1) == 0){
    	cerr<<"Number of processes should be greater than 1 and (num of processes-1) should be divisor of 1000 \n";
    	exit(1);
    }

    // Get the rank of the process
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Status status;

    // Get the name of the processor
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);

    //Define a new MPI datatype for struct Point3f
	int num_members = 3;    			// Number of members in struct
	int lengths[num_members] = { 1, 1, 1};			// Length of each member in struct
	MPI_Aint offsets[num_members] = {0, sizeof(float), 2*sizeof(float)};
	MPI_Datatype types[num_members] = { MPI_FLOAT, MPI_FLOAT, MPI_FLOAT};
	MPI_Datatype Point3f_type;
	MPI_Type_struct(num_members, lengths, offsets, types, &Point3f_type);
	MPI_Type_commit(&Point3f_type);

	//Define a new MPI datatype for struct DMatch
	int num_membersD = 3;    							// Number of members in struct
	int lengthsD[num_membersD] = { 1, 1, 1};			// Length of each member in struct
	MPI_Aint offsetsD[num_membersD] = {0, sizeof(int), 2*sizeof(int)};
	MPI_Datatype typesD[num_membersD] = { MPI_INT, MPI_INT, MPI_FLOAT};
	MPI_Datatype DMatch_type;
	MPI_Type_struct(num_membersD, lengthsD, offsetsD, typesD, &DMatch_type);
	MPI_Type_commit(&DMatch_type);

    double time = 0;
    time_best result;
    std::vector<KeyPoint> keypoints_1, keypoints_2;
    std::vector< DMatch > good_matches;
    Mat img_1, img_2, img_matches;
    int chunk_size = 1000/(world_size - 1);
    std::vector< DMatch > matches;
    int size_Bcast;

    if(my_rank == 0){
        if( argc != 3 ){ 
            readme();
            return -1; 
        }

        img_1 = imread( argv[1], IMREAD_GRAYSCALE );
        img_2 = imread( argv[2], IMREAD_GRAYSCALE );

        if( !img_1.data || !img_2.data ){ 
            std::cout<< " --(!) Error reading images " << std::endl; return -1; 
        }

        //-- Step 1: Detect the keypoints using SIFT Detector, compute the descriptors
        Ptr<SIFT> detector = SIFT::create();
        Mat descriptors_1, descriptors_2;
        detector->detectAndCompute( img_1, Mat(), keypoints_1, descriptors_1 );
        detector->detectAndCompute( img_2, Mat(), keypoints_2, descriptors_2 );

        //-- Step 2: Matching descriptor vectors using FLANN matcher
        FlannBasedMatcher matcher;
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
        for( int i = 0; i < descriptors_1.rows; i++ ){ 
            if( matches[i].distance <= max(6*min_dist, 0.02) ){ 
                good_matches.push_back( matches[i]); 
            }
        }
        // DMatch *good_matches_a = &good_matches[0];			//converting vector to array
        
        drawMatches( img_1, keypoints_1, img_2, keypoints_2, good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
                 vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
        //-- Show detected matches
        imshow( "Good Matches", img_matches );

        //for( int i = 0; i < (int)good_matches.size(); i++ )
        //{ 
        //printf( "-- Good Match [%d] Keypoint 1: %d  -- Keypoint 2: %d  \n", i, good_matches[i].queryIdx, good_matches[i].trainIdx ); 
        //}
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

        for(int k = 0; k< 2; k++){
            rand_idx_src.clear();
            rand_idx_dst.clear();
            for(int j = 0; j<3; j++){
                do{
                    num = rand()%N;
                }while (contains(rand_idx_src, num));

                rand_idx_src.push_back(num);
                src_vec.push_back(src_pts[num]);
                //cout<<"source random index ["<<j<<"]:"<< num << endl;

                do{
                    num = rand()%N;
                }while (contains(rand_idx_dst, num));

                rand_idx_dst.push_back(num);
                dst_vec.push_back(dst_pts[num]);
                //cout<<"destination random index ["<<j<<"]:"<< num << endl;
            }
        }

        //Process 0 will now divide the src and dst vector and points of good matches between all the process
		//Converting vectors to arrays before broadcasting
		Point3f *src_vec_a = &src_vec[0];
		Point3f *dst_vec_a = &dst_vec[0];
		Point3f *src_pts_a = &src_pts[0];
		Point3f *dst_pts_a = &dst_pts[0];
		int i,j;
		size_Bcast = (int)good_matches.size();
        DMatch_new good_matches_struct[size_Bcast];
        for( int i = 0; i < size_Bcast; i++){
        	good_matches_struct[i].queryIdx = good_matches[i].queryIdx;
        	good_matches_struct[i].trainIdx = good_matches[i].trainIdx;
        	good_matches_struct[i].distance = good_matches[i].distance;        	
		}
    
		cout<<"my_rank = "<<my_rank<<" src_vec = "<<src_vec_a[0].x<<endl<<std::flush;

		for(i = 0; i<world_size-1; i++){
			MPI_Send(&src_vec_a[i*chunk_size], chunk_size, Point3f_type, i+1, 0, MPI_COMM_WORLD);
			MPI_Send(&dst_vec_a[i*chunk_size], chunk_size, Point3f_type, i+1, 1, MPI_COMM_WORLD);
			MPI_Send(&src_pts_a[i*chunk_size], chunk_size, Point3f_type, i+1, 2, MPI_COMM_WORLD);
			MPI_Send(&dst_pts_a[i*chunk_size], chunk_size, Point3f_type, i+1, 3, MPI_COMM_WORLD);
			MPI_Send(&size_Bcast, 1, MPI_INT, i+1, 99, MPI_COMM_WORLD);
			MPI_Send(&good_matches_struct[0], size_Bcast, DMatch_type, i+1, 100, MPI_COMM_WORLD);
		}

    }
    else{
        time = 0;
        Point3f src_vec_a[chunk_size], dst_vec_a[chunk_size], src_pts_a[chunk_size], dst_pts_a[chunk_size];
        MPI_Recv(&(src_vec_a[0]), chunk_size, Point3f_type, 0, 0, MPI_COMM_WORLD, &status);
        MPI_Recv(&(dst_vec_a[0]), chunk_size, Point3f_type, 0, 1, MPI_COMM_WORLD, &status);
        MPI_Recv(&(src_pts_a[0]), chunk_size, Point3f_type, 0, 2, MPI_COMM_WORLD, &status);
        MPI_Recv(&(dst_pts_a[0]), chunk_size, Point3f_type, 0, 3, MPI_COMM_WORLD, &status);
        cout<<"my_rank = "<<my_rank<<" src_vec = "<<src_vec_a[0].x<<endl<<std::flush;

        int size_Bcast;
        MPI_Recv(&size_Bcast, 1, MPI_INT, 0, 99, MPI_COMM_WORLD, &status);
        cout<<"my_rank = "<<my_rank<<" sizeBcast = "<<size_Bcast<<endl<<std::flush;

        DMatch_new good_matches_struct[size_Bcast];
        MPI_Recv(&(good_matches_struct[0]), size_Bcast, DMatch_type, 0, 100, MPI_COMM_WORLD, &status);
        cout<<"my_rank = "<<my_rank<<" good_matches = "<<good_matches_struct[0].queryIdx<<endl<<std::flush;

        std::vector<Point3f> src_vec_new(src_vec_a, src_vec_a + sizeof(src_vec_a)/sizeof(src_vec_a[0]));
        std::vector<Point3f> dst_vec_new(dst_vec_a, dst_vec_a + sizeof(dst_vec_a)/sizeof(dst_vec_a[0]));
        std::vector<Point3f> src_pts_new(src_pts_a, src_pts_a + sizeof(src_pts_a)/sizeof(src_pts_a[0]));
        std::vector<Point3f> dst_pts_new(dst_pts_a, dst_pts_a + sizeof(dst_pts_a)/sizeof(dst_pts_a[0]));
        std::vector<DMatch_new> good_matches_new(good_matches_struct, good_matches_struct + sizeof(good_matches_struct)/sizeof(good_matches_struct[0]));
        printf("MY rank %d\n", my_rank);

        result = RANSAC_parallel(src_vec_new, dst_vec_new, src_pts_new, dst_pts_new, good_matches_new, chunk_size);
        printf("Rank : %d, time = %f\n", my_rank, result.time );
    }
    
   
    //RUNNING EXPERIMENTS
   
    // printf("Average time = %f\n", time/nb_experiments);

    // std::vector<DMatch> best_matches = result.best_matches;

    // //Checking for serial and parallel implementation
    // for(int i = 0; i < 5; i++){
    //     cout<<keypoints_1[best_matches[i].queryIdx].pt.x<<endl;
    // }

    // drawMatches( img_1, keypoints_1, img_2, keypoints_2, best_matches, img_matches, Scalar::all(-1), Scalar::all(-1),vector<char>(), 
    // DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
    // //-- Show detected best matches
    // //imshow( "Best Matches", img_matches );
    // imwrite("Result.jpg", img_matches);
    //waitKey(0);
    
    MPI_Finalize();

    return 0;

}

time_best RANSAC_parallel(std::vector<Point3f> src_vec, std::vector<Point3f> dst_vec, 
    std::vector<Point3f> src_pts, std::vector<Point3f> dst_pts, std::vector<DMatch_new> good_matches, int chunk_size){

    // --- RANSAC Algorithm Serial
    int N = (int)good_matches.size();
    cout<<"good_matches size = "<<N<<endl<<std::flush;
    cout<<"src_vec size = "<<(int)src_vec.size()<<endl<<std::flush;
    std::vector<Point3f> src_random, dst_random;
    int outliers;
    int past_outliers = N +1;
    std::vector<int> idx_remove, Best_idx_remove;
    std::vector< DMatch_new > best_matches = good_matches;
    Mat H;

    register unsigned int i,j,k;
    double start_time = omp_get_wtime();

    #pragma omp parallel for schedule(dynamic) private(i, j, k, idx_remove, H, src_random, dst_random, outliers)
    for(k =0; k<chunk_size; k++){
        idx_remove.clear();
        src_random.clear();
        dst_random.clear();
        outliers = 0;

        // #pragma omp critical
        // {
            for(j = 0; j<3; j++){
            	// printf("hi");
                src_random.push_back(src_vec.back());
                src_vec.pop_back();
                dst_random.push_back(dst_vec.back());
                dst_vec.pop_back();
            }
        // }
        // printf("hi1");
        H = findHomography(src_random, dst_random);
        // printf("hi2");
        for(i = 0 ; i<N ; i++){
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
        	cout<<k<<endl;
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
