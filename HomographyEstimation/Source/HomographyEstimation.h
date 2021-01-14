#pragma once
#ifndef __HOMOGRAPHIES__
#define __HOMOGRAPHIES__

#include <iostream>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;

class Homographies {
public:
	Homographies(Mat& im1, Mat& im2);
	void HomographyMenu();

private:
	Mat im1, im2;
	vector < pair<Point2f, Point2f> > pointPairs;

	typedef struct normalizedData {
		Mat T2D_x;
		Mat T2D_xDash;
		vector<Point2f> newPts2D_x;
		vector<Point2f> newPts2D_xDash;
	}normalized;

	void PlanarHomography();
	void PanoramicImaging();
	void FeatureMatching();
	Mat RobustFitting();
	void TransformImage(Mat origImg, Mat& newImage, Mat tr, bool isPerspective);
	void SaveMatchingPointPairs(vector<KeyPoint>& matched1, vector<KeyPoint>& matched2);
	void NormaliseData(vector<pair<Point2f, Point2f> >& pointPairs, normalized& norm);
	size_t GetIterationNumber(const double &inlierRatio_, const double &confidence_, const size_t &sampleSize_);
	Mat CalcHomography(vector<pair<Point2f, Point2f> >& pointPairs);
};

#endif // !__HOMOGRAPHIES__

