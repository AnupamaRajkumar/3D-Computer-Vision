// CV_Practise_RANSAC.cpp : Defines the entry point for the console application.
//


#include "StructureFromMotion.h"



int main(int argc, char** argv)
{
	if (argc < 4) {
		std::cerr << "Looking for 3 inputs" << std::endl;
		std::cerr << "Enter two images for stereo reconstruction and the corresponding matching feature points" << std::endl;
		exit(0);
	}

	// Load images
	cv::Mat image1, image2;
	image1 = cv::imread(argv[1], 1);
	image2 = cv::imread(argv[2], 1);
	char* matchedFile;
	matchedFile = argv[3];
	cout << matchedFile << endl;

	SFM sfm(image1, image2, matchedFile);

	cv::waitKey(0);

	return 0;
}

