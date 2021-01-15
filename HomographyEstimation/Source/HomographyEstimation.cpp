#include "HomographyEstimation.h"

Homographies::Homographies(Mat& im1, Mat& im2, string folderName) {
	this->im1 = im1.clone();
	this->im2 = im2.clone();
	this->folderName = folderName;

	/*Invoke the homographies menu*/
	this->HomographyMenu();
}

void Homographies::HomographyMenu() {
	int choice = 1;

	cout << "Linear Homography Estimation Menu" << endl;
	cout << "1. Transformation of Planar Patterns" << endl;
	cout << "2. Panoramic Imaging" << endl;
	cout << "Please enter your choice (1/2)" << endl;
	cin >> choice;
	switch (choice)
	{
	case 1:
		this->PlanarHomography();
		break;
	case 2:
		this->PanoramicImaging();
		break;
	default:
		cerr << "Enter valid choice (1/2)" << endl;
		break;
	}
}

void Homographies::PlanarHomography() {
	/*step 1 : Perform Feature Matching*/
	this->FeatureMatching();
	/*step 2 : Robustify these points using standard RANSAC*/
	Mat bestHomography;
	bestHomography = this->RobustFitting();
	cout << "Best homography found" << endl;
	/*step 3 : Apply linear homography estimation*/
	Mat transformedImage = Mat::zeros(this->im1.size().height, this->im1.size().width, this->im1.type());
	this->TransformImage(this->im2, transformedImage, bestHomography, true, false);
	imwrite("LinearHomography.png", transformedImage);
	imshow("Linear Homography", transformedImage);
}

void Homographies::PanoramicImaging(){
	/*step 1 : Load panoramic images*/
	vector<Mat> panoramicImages;
	this->LoadPanoramicImages(panoramicImages);
	/*step 2 : Perform Feature Matching*/
	this->FeatureMatching();
	/*step 3 : Robustify these points using standard RANSAC*/
	Mat bestHomography;
	bestHomography = this->RobustFitting();
	cout << "Best homography found" << endl;
	/*step 3: Apply linear homography estimation and stitch the images together*/
	Mat transformedImage = Mat::zeros(1.5 * this->im1.size().height, 1.5 * this->im1.size().width, this->im1.type());
	this->TransformImage(this->im2, transformedImage, Mat::eye(3, 3, CV_32F), true, true);
	this->TransformImage(this->im1, transformedImage, bestHomography, true, true);

	imwrite("PanoramicStitching.png", transformedImage);
	imshow("PanoramicStitching", transformedImage);

}

/*Akaze feature tracking referred from https://docs.opencv.org/3.4/db/d70/tutorial_akaze_matching.html */
void Homographies::FeatureMatching() {
	vector<KeyPoint> kpts1, kpts2;
	Mat desc1, desc2;
	float nn_match_ratio = 0.8f;

	Ptr<AKAZE> akaze = AKAZE::create();
	akaze->detectAndCompute(this->im1, noArray(), kpts1, desc1);
	akaze->detectAndCompute(this->im2, noArray(), kpts2, desc2);

	BFMatcher matcher(NORM_HAMMING);
	vector<vector<DMatch>> nn_matches;
	matcher.knnMatch(desc1, desc2, nn_matches, 2);

	vector<KeyPoint> matched1, matched2;
	for (size_t i = 0; i < nn_matches.size(); i++) {
		DMatch first = nn_matches[i][0];
		float dist1 = nn_matches[i][0].distance;
		float dist2 = nn_matches[i][1].distance;

		if (dist1 < nn_match_ratio * dist2) {
			matched1.push_back(kpts1[first.queryIdx]);
			matched2.push_back(kpts2[first.trainIdx]);
		}
	}
	this->SaveMatchingPointPairs(matched1, matched2);
}

void Homographies::SaveMatchingPointPairs(vector<KeyPoint>& matched1, vector<KeyPoint>& matched2) {

	for (size_t i = 0; i < matched1.size(); i++) {
		pair<Point2f, Point2f> points;
		points.first = matched1[i].pt;
		points.second = matched2[i].pt;
		this->pointPairs.emplace_back(points);
	}
}

Mat Homographies::RobustFitting() {
	int numberOfIterations = 100;
	int maxNumOfIterations = numberOfIterations;
	double confidence = 0.999999;
	int minInlierNum = 170;
	int threshold = 5;
	int numOfSamples = 4;
	int iterationNum = 0;
	vector<int> inliers;
	vector<int> bestInliers;
	Mat bestHomography;

	inliers.reserve(pointPairs.size());
	bestInliers.reserve(pointPairs.size());

	while (iterationNum++ < maxNumOfIterations) {
		/*choose 4 random points from the point pairs*/
		vector < pair<Point2f, Point2f> > selectedPts;
		for (size_t i = 0; i < numOfSamples; i++) {
			// Generate a random index between [0, pointNumber]
			int idx;
			idx = round((pointPairs.size() - 1) * static_cast<double>(rand()) / static_cast<double>(RAND_MAX));
			selectedPts.emplace_back(pointPairs[idx]);
		}
		/*fit these 4 points to homography model*/
		Mat H;
		H = this->CalcHomography(selectedPts);

		/*Count the number of inliers ie. points closer to the threshold*/
		inliers.clear();
		for (size_t pointIdx = 0; pointIdx < pointPairs.size(); ++pointIdx)
		{
			Point2f point_x = pointPairs[pointIdx].first;
			Point2f point_xDash = pointPairs[pointIdx].second;
			Mat pt(3, 1, CV_32F);
			pt.at<float>(0, 0) = point_x.x;
			pt.at<float>(1, 0) = point_x.y;
			pt.at<float>(2, 0) = 1.0;
			Mat ptTransformed = H * pt;
			ptTransformed = (1.0 / ptTransformed.at<float>(2, 0)) * ptTransformed;
			double distance = 0.;
			double cartDistX = (round(ptTransformed.at<float>(0, 0)) - point_xDash.x);
			double cartDistY = (round(ptTransformed.at<float>(1, 0)) - point_xDash.y);
			distance = sqrt(cartDistX * cartDistX + cartDistY * cartDistY);
			if (distance < threshold)
			{
				inliers.emplace_back(pointIdx);
			}
		}

		/* Store the inlier number and the line parameters if it is better than the previous best. */
		if (inliers.size() > bestInliers.size())
		{
			bestInliers.swap(inliers);
			inliers.clear();
			inliers.resize(0);

			bestHomography = H.clone();

			// Update the maximum iteration number
			maxNumOfIterations = GetIterationNumber(
				static_cast<double>(bestInliers.size()) / static_cast<double>(pointPairs.size()),
				confidence,
				numOfSamples);

			//printf("Inlier number = %d\n", bestInliers.size());
			cout << "Inlier number :" << bestInliers.size() << " max iterations :" << maxNumOfIterations << endl;
		}
	}
	return bestHomography;
}

size_t Homographies::GetIterationNumber(const double &inlierRatio_, const double &confidence_, const size_t &sampleSize_)
{
	double a =
		log(1.0 - confidence_);
	double b =
		log(1.0 - std::pow(inlierRatio_, sampleSize_));

	if (abs(b) < std::numeric_limits<double>::epsilon())
		return std::numeric_limits<size_t>::max();

	return a / b;
}

/*Data normalization should be applied before applying homographies*/
void Homographies::NormaliseData(vector<pair<Point2f, Point2f> >& pointPairs, normalized& norm) {
	int ptsNum = pointPairs.size();

	//calculate means (they will be the center of coordinate systems)
	float mean1x = 0.0, mean1y = 0.0, mean2x = 0.0, mean2y = 0.0;
	for (int i = 0; i < ptsNum; i++) {
		/*mean of points in x in order to move the centroid to the origin*/
		mean1x += pointPairs[i].first.x;			
		mean1y += pointPairs[i].first.y;

		/*mean of points in x_dash in order to move the centroid to the origin*/
		mean2x += pointPairs[i].second.x;
		mean2y += pointPairs[i].second.y;
	
	}
	mean1x /= ptsNum;
	mean1y /= ptsNum;

	mean2x /= ptsNum;
	mean2y /= ptsNum;

	float spread1x = 0.0, spread1y = 0.0, spread2x = 0.0, spread2y = 0.0;

	for (int i = 0; i < ptsNum; i++) {
		spread1x += (pointPairs[i].first.x - mean1x)*(pointPairs[i].first.x - mean1x);
		spread1y += (pointPairs[i].first.y - mean1y)*(pointPairs[i].first.y - mean1y);

		spread2x += (pointPairs[i].second.x - mean2x)*(pointPairs[i].second.x - mean1x);
		spread2y += (pointPairs[i].second.y - mean2y)*(pointPairs[i].second.y - mean2y);
	}

	spread1x /= ptsNum;
	spread1y /= ptsNum;

	spread2x /= ptsNum;
	spread2y /= ptsNum;

	Mat offs1 = Mat::eye(3, 3, CV_32F);
	Mat offs2 = offs1.clone();
	Mat scale1 = Mat::eye(3, 3, CV_32F);
	Mat scale2 = scale1.clone();

	offs1.at<float>(0, 2) = -mean1x;
	offs1.at<float>(1, 2) = -mean1y;

	offs2.at<float>(0, 2) = -mean2x;
	offs2.at<float>(1, 2) = -mean2y;

	scale1.at<float>(0, 0) = sqrt(2) / sqrt(spread1x);
	scale1.at<float>(1, 1) = sqrt(2) / sqrt(spread1y);

	scale2.at<float>(0, 0) = sqrt(2) / sqrt(spread2x);
	scale2.at<float>(1, 1) = sqrt(2) / sqrt(spread2y);

	norm.T2D_x = scale1 * offs1;
	norm.T2D_xDash = scale2 * offs2;

	for (int i = 0; i < ptsNum; i++) {
		Point2f p2D_x;
		Point2f p2D_xDash;

		p2D_x.x = sqrt(2) * (pointPairs[i].first.x - mean1x) / sqrt(spread1x);
		p2D_x.y = sqrt(2) * (pointPairs[i].first.y - mean1y) / sqrt(spread1y);

		p2D_xDash.x = sqrt(2) * (pointPairs[i].second.x - mean2x) / sqrt(spread2x);
		p2D_xDash.y = sqrt(2) * (pointPairs[i].second.y - mean2y) / sqrt(spread2y);

		norm.newPts2D_x.push_back(p2D_x);
		norm.newPts2D_xDash.push_back(p2D_xDash);
	}

}

Mat Homographies::CalcHomography(vector<pair<Point2f, Point2f> >& pointPairs) {
	normalized norm;
	this->NormaliseData(pointPairs, norm);

	const int ptsNum = pointPairs.size();
	Mat A(2 * ptsNum, 9, CV_32F);
	for (int i = 0; i < ptsNum; i++) {

		float u1 = norm.newPts2D_x[i].x;
		float v1 = norm.newPts2D_x[i].y;

		float u2 = norm.newPts2D_xDash[i].x;
		float v2 = norm.newPts2D_xDash[i].y;

		A.at<float>(2 * i, 0) = u1;
		A.at<float>(2 * i, 1) = v1;
		A.at<float>(2 * i, 2) = 1.0f;
		A.at<float>(2 * i, 3) = 0.0f;
		A.at<float>(2 * i, 4) = 0.0f;
		A.at<float>(2 * i, 5) = 0.0f;
		A.at<float>(2 * i, 6) = -u2 * u1;
		A.at<float>(2 * i, 7) = -u2 * v1;
		A.at<float>(2 * i, 8) = -u2;

		A.at<float>(2 * i + 1, 0) = 0.0f;
		A.at<float>(2 * i + 1, 1) = 0.0f;
		A.at<float>(2 * i + 1, 2) = 0.0f;
		A.at<float>(2 * i + 1, 3) = u1;
		A.at<float>(2 * i + 1, 4) = v1;
		A.at<float>(2 * i + 1, 5) = 1.0f;
		A.at<float>(2 * i + 1, 6) = -v2 * u1;
		A.at<float>(2 * i + 1, 7) = -v2 * v1;
		A.at<float>(2 * i + 1, 8) = -v2;

	}

	Mat eVecs(9, 9, CV_32F), eVals(9, 9, CV_32F);
	eigen(A.t() * A, eVals, eVecs);

	Mat H(3, 3, CV_32F);
	for (int i = 0; i < 9; i++) H.at<float>(i / 3, i % 3) = eVecs.at<float>(8, i);

	// Data normalization
	H = norm.T2D_xDash.inv()*H*norm.T2D_x;
	//Normalize:
	H = H * (1.0 / H.at<float>(2, 2));

	return H;
}

void Homographies::TransformImage(Mat origImg, Mat& newImage, Mat tr, bool isPerspective, bool isInvTrNeeded) {
	const int WIDTH = origImg.cols;
	const int HEIGHT = origImg.rows;

	const int newWIDTH = newImage.cols;
	const int newHEIGHT = newImage.rows;

	for (int x = 0; x < newWIDTH; x++) for (int y = 0; y < newHEIGHT; y++) {
		Mat pt(3, 1, CV_32F);
		pt.at<float>(0, 0) = x;
		pt.at<float>(1, 0) = y;
		pt.at<float>(2, 0) = 1.0;

		Mat ptTransformed;
		if (isInvTrNeeded)
			ptTransformed = tr.inv() * pt;
		else
			ptTransformed = tr * pt;

		if (isPerspective) ptTransformed = (1.0 / ptTransformed.at<float>(2, 0)) * ptTransformed;

		int newX = round(ptTransformed.at<float>(0, 0));
		int newY = round(ptTransformed.at<float>(1, 0));

		if ((newX >= 0) && (newX < WIDTH) && (newY >= 0) && (newY < HEIGHT))
			newImage.at<Vec3b>(y, x) = origImg.at<Vec3b>(newY, newX);
	}
}

void Homographies::LoadPanoramicImages(vector<Mat>& panoramicImages) {
	fs::recursive_directory_iterator iter(this->folderName);
	fs::recursive_directory_iterator end;
	while (iter != end) {
		Mat img = imread(iter->path().string());
		if (!img.data) {
			cout << "ERROR: Cannot find labelled image" << endl;
			cout << "Press enter to exit..." << endl;
			cin.get();
			exit(0);
		}
		img.convertTo(img, CV_32FC1);
		panoramicImages.push_back(img);

		error_code ec;
		iter.increment(ec);
		if (ec) {
			std::cerr << "Error while accessing:" << iter->path().string() << "::" << ec.message() << "\n";
		}
	}
}