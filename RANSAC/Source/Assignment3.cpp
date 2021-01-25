/*
1. Read input image
2. Apply canny edge detector to get an edge image - edge points are considered as 2D points
3. Draw line if at least 200 points (inliers) support the candidate line
4. Iteration number is a parameter for the application
5. Draw fitted line to the original image by red
*/

#include "LineFitting.h"
#include "OptimisedRANSAC.h"

int main(int argc, char** argv)
{
	if (argc < 3) {
		std::cerr << "Input format : Input image Lidar scan" << endl;
		return EXIT_FAILURE;
	}

	int choice = 1;
	cout << "-----------Robust estimation menu----------\n";
	cout << "1. Robust line fitting using RANSAC" << endl;
	cout << "2. Robust plane fitting using LO RANSAC" << endl;
	cout << "Please enter your choice (1/2):" << endl;
	cin >> choice;
	Mat img = imread(argv[1]);
	int iterations = 100;
	int option = 1;


	cout << "Enter the number of iterations desired" << endl;
	cin >> iterations;
	if (!img.data) {
		std::cerr << "No image data" << endl;
		return EXIT_FAILURE;
	}	
	
	LineFit line(img, iterations);
	loRANSAC loransac(argv[2], iterations);
	switch (choice) {
		case 1:	
			line.RobustFitting();
			break;
		case 2:
			cout << "Optimisation method menu:" << endl;
			cout << "1. Least Square Optimisation:" << endl;
			cout << "2. Iterative Least Square Optimisation:" << endl;
			cout << "3. No optimisation" << endl;
			cout << "Enter your choice (1/2/3):" << endl;
			cin >> option;
			loransac.LocallyOptimisedRANSAC(option);
			break;
		default:
			cout << "Enter a valid choice!! (1 or 2)" << endl;
			break;
	}


		
	cv::waitKey(0);
	return 0;
}

