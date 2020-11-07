#pragma once
#ifndef __LINEFIT__
#define __LINEFIT__

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;

class LineFit {
public:
	LineFit(Mat& img, int iterations);
	void RobustFitting();
	
private:
	Mat img;
	int iterations;

	Mat EdgeDetector();
	void FitLineRANSAC(const vector<Point2d> &points_, const vector<int> &mask_, vector<int> &inliers_, Mat &line_,
						double threshold_, double confidence_, int maximum_iteration_number_, Mat *image_);

	void SequentialRANSAC(vector<Point2d> &points, vector<vector<int>> &inliers, vector<Mat> &lines, double threshold,
							double confidence, int iteration_number, int minInlinerNum, 
							int lineNumber = numeric_limits<int>::max(), Mat *image = nullptr);
	size_t GetIterationNumber(const double &inlierRatio_, const double &confidence_, const size_t &sampleSize_);

	void FitLineLSQ(const vector<Point2d> * const points, vector<int> &inliers, Mat &line);
	vector<Point2d> GeneratePoints(Mat& edge);
};
#endif // !__LINEFIT__

