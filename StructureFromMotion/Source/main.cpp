// CV_Practise_RANSAC.cpp : Defines the entry point for the console application.
//


#include "StructureFromMotion.h"



int main(int argc, char** argv)
{
	if (argc < 3) {
		std::cerr << "Looking for 2 inputs" << std::endl;
		std::cerr << "Enter two images for stereo reconstruction" << std::endl;
		exit(0);
	}

	// Load images
	cv::Mat image1, image2;
	image1 = cv::imread(argv[1], 1);
	image2 = cv::imread(argv[2], 1);

	SFM sfm(image1, image2);

	cv::waitKey(0);

	return 0;
}

