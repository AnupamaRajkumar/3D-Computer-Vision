#pragma once
#ifndef __LORANSAC__
#define __LORANSAC__

#include <iostream>
#include <opencv2/opencv.hpp>
#include <fstream>

using namespace cv;
using namespace std;

class loRANSAC {
public:
	loRANSAC(char* fileName, int iterations);
	void LocallyOptimisedRANSAC(int option);

private:
	char* fileName;
	int iterations;
	typedef struct points {
		Point3d point;
		int red;
		int green;
		int blue;
	} allPoints;

	void FitLoRANSAC(const vector<allPoints>& points_, int maximum_iteration_number_, double threshold_, 
					double loThreshold, double confidence_, vector<int>& inliers_, Mat& plane_, int option);
	//void FitPlaneLSQ(vector<allPoints>& points, vector<int> &inliers, Mat &plane);
	void FitPlaneLSQ(const vector<allPoints>& points_, vector<int>& inliers_, Mat& bestPlane);
	void LoadData(vector<allPoints>& points);
	void WriteDataPoints(vector<allPoints>& points);
	size_t GetIterationNumber(const double &inlierRatio_, const double &confidence_, const size_t &sampleSize_);
};

#endif // !__LORANSAC__

