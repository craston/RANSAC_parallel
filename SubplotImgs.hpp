#include <opencv2/opencv.hpp>

#include <stdio.h>
//#include <stdarg.h>

using namespace cv;
using namespace std;

//returns an image containg all images

Mat SubPlot(vector<Mat> imgs, int ChangeSize=1 ) {

int size;
int i;
int m, n;
int x, y;

// w - Maximum number of images in a row
// h - Maximum number of images in a column
int w, h;

// scale - How much we have to resize the image
float scale;
int max;

int nArgs = imgs.size();
// If the number of arguments is lesser than 0 or greater than 12
// return without displaying
if(nArgs <= 0) {
    printf("Number of arguments too small....\n");
    return Mat::zeros(Size(100, 60), CV_8UC3);
}
else if(nArgs > 14) {
    printf("Number of arguments too large, can only handle maximally 12 images at a time ...\n");
    return Mat::zeros(Size(100, 60), CV_8UC3);
}
// Determine the size of the image,
// and the number of rows/cols
// from number of arguments
else if (nArgs == 1) {
    w = h = 1;
    size = 300;
}
else if (nArgs == 2) {
    w = 2; h = 1;
    size = 300;
}
else if (nArgs == 3 || nArgs == 4) {
    w = 2; h = 2;
    size = 300;
}
else if (nArgs == 5 || nArgs == 6) {
    w = 3; h = 2;
    size = 200;
}
else if (nArgs == 7 || nArgs == 8) {
    w = 4; h = 2;
    size = 200;
}
else {
    w = 4; h = 3;
    size = 150;
}

// Create a new 3 channel image
//Mat DispImage = Mat::zeros(Size(100 + size*w, 60 + size*h), CV_8UC3);
// Used to get the arguments passed


x = imgs[0].cols;
y = imgs[0].rows;
assert(x!=0 && y != 0);
// Find whether height or width is greater in order to resize the image
max = (x > y)? x: y;

// Find the scaling factor to resize the image
scale = (float) ( (float) max / size );
int Y;
if(nArgs>3) Y = (int)( y/scale - 200);
else Y = (int)( y/scale);

Mat DispImage((int)max + Y,(int)max +(int)( x/scale ), CV_8UC1, Scalar(128));

// Loop for nArgs number of arguments
for (i = 0, m = 20, n = 20; i < nArgs; i++, m += (20 + size)) {
    // Get the Pointer to the IplImage
    Mat img = imgs[i];

    // Check whether it is NULL or not
    // If it is NULL, release the image, and return
    if(img.empty()) {
        printf("Invalid arguments");
        return Mat::zeros(Size(100, 60), CV_8UC3);
    }

    // Used to Align the images
    if( i % w == 0 && m!= 20) {
        m = 20;
        n+= 20 + size;
    }

    // Set the image ROI to display the current image
    // Resize the input image and copy it to the Single Big Image

    if(ChangeSize)
    {
        Rect ROI(m, n, (int)( x/scale ), (int)( y/scale ));
        Mat temp; 
        resize(img,temp, Size(ROI.width, ROI.height));
        temp.copyTo(DispImage(ROI));
    }
    else{
        Rect ROI(m, n, (int)( x/scale ), (int)( y/scale ));
        img.copyTo(DispImage(ROI));
    }
    //img.copyTo(DispImage);
}

return DispImage;
}
/*
#define CV_8U   0
#define CV_8S   1 
#define CV_16U  2
#define CV_16S  3
#define CV_32S  4
#define CV_32F  5
#define CV_64F  6
*/
///Useful function to get the type of MAT
string type2str(int type) {
  string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }
  return r;
}