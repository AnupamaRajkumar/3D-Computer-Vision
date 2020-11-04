/*
1. Read input image
2. Apply canny edge detector to get an edge image - edge points are considered as 2D points
3. Draw line if at least 200 points (inliers) support the candidate line
4. Iteration number is a parameter for the application
5. Draw fitted line to the original image by red
*/


#include "LineFitting.h"

int main(int argc, char** argv)
{
	if (argc < 2) {
		std::cerr << "Input image not found" << endl;
		return EXIT_FAILURE;
	}

	Mat img = imread(argv[1]);

	if (!img.data) {
		std::cerr << "No image data" << endl;
		return EXIT_FAILURE;
	}

	LineFit line(img);
	line.RobustFitting();
		
	cv::waitKey(0);
	return 0;
}

