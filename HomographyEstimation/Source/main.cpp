/* Homography transformation:
1. Perform feature matching on two images
2. Implement normalized linear homography estimation
3. Robustify the method using RANSAC - Apply robust estimation to at least 3 images
4. Stitch panoramic images
*/


//#if 0
#include "HomographyEstimation.h"

int main(int argc, char** argv)
{
	if (argc < 4) {
		cerr << "Looking for 3 inputs" << endl;
		cerr << "Enter two images for homography estimation and folder name containing images for panoramic stitching" << endl;
		exit(0);
	}
	
	Mat im1, im2;
	im1 = imread(argv[1], 1);
	im2 = imread(argv[2], 1);

	string folderName;
	folderName = argv[3];

	/*resize the image to 50% of it's original size so as to reduce
	the size of the keypoint array*/
	resize(im1, im1, cv::Size(), 0.5, 0.5);
	resize(im2, im2, cv::Size(), 0.5, 0.5);

	Homographies homography(im1, im2, folderName);

	waitKey(0);
	return 0;
}

//#endif

