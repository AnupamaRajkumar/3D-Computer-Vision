#include "LineFitting.h"

LineFit::LineFit(Mat& img, int iterations) {
	this->img = img;
	this->iterations = iterations;
}

Mat LineFit::EdgeDetector() {
	Mat edgeImg = Mat::zeros(img.size(), CV_8UC1);
	Mat binImg = edgeImg.clone();

	/*step 1: Filtering to smooth out noise*/
	int kSize = 3;
	Mat kernel = Mat(kSize, kSize, img.type());
	Mat filtImg = Mat::zeros(img.size(), img.type());
	/*converting image to grayscale*/
	cvtColor(img, filtImg, COLOR_BGR2GRAY);
	//GaussianBlur(filtImg, filtImg, kernel.size(), 0, 0);

	/*step 2 : Thresholding*/
	double maxVal = 255.;
	double thresh = 0.;
	threshold(filtImg, binImg, thresh, maxVal, THRESH_BINARY + THRESH_OTSU);

	/*Step 3 : Edge filtering*/
	Canny(binImg, edgeImg, 0, 2, 3);

	imwrite("Edge.png", edgeImg);
	return edgeImg;
}
vector<Point2d> LineFit::GeneratePoints(Mat& edge) {
	vector<Point2d> points;
	Mat locations;
	findNonZero(edge, locations);
	for (int i = 0; i < locations.rows; i++) {
		for (int j = 0; j < locations.cols; j++) {
			Point2d point = locations.at<Point>(i, j);
			points.push_back(point);
		}
	}
	return points;
}

void LineFit::RobustFitting() {
	/*Generate an edge image*/
	Mat edge = this->EdgeDetector();
	Mat fitImg = img.clone();
	/*Calculate points from the edge image*/
	vector<Point2d> points = this->GeneratePoints(edge);
	cout << points.size() << endl;
	/*Fit the line robustly*/
	vector<vector<int>> inliers;
	vector<Mat> bestLines;
	int threshold = 10;
	int numOfIterations = iterations;
	int minInlierNum = 700;
	this->SequentialRANSAC(points, inliers, bestLines, threshold, 0.999999, numOfIterations, minInlierNum);

	cout << "Number of lines found:" << bestLines.size() << endl;

	for (int lineIdx = 0; lineIdx < bestLines.size(); lineIdx++) {

		// Draw the line from Least Squares Fitting
		const double &a2 = bestLines[lineIdx].at<double>(0);
		const double &b2 = bestLines[lineIdx].at<double>(1);
		const double &c2 = bestLines[lineIdx].at<double>(2);

		// Draw the 2D line
		cv::line(fitImg,
			Point2d(0, -c2 / b2),
			Point2d(fitImg.cols, (-a2 * fitImg.cols - c2) / b2),
			cv::Scalar(0, 0, 255),
			2);

		// Calculate the error or the Least Squares Fitting line
		double averageError = 0.0;
		for (const auto &inlierIdx : inliers[lineIdx])
		{
			double distance = abs(a2 * points[inlierIdx].x + b2 * points[inlierIdx].y + c2);
			averageError += distance;
		}
		averageError /= inliers[lineIdx].size();

		cout << "Average RANSAC error for line " << lineIdx + 1 << " is : " << averageError << endl;
	}
	imshow("Final result", fitImg);
	imwrite("RobustFitting.png", fitImg);
}

void LineFit::SequentialRANSAC(vector<Point2d> &points, vector<vector<int>> &inliers, vector<Mat> &lines, double threshold,
	double confidence, int iteration_number, int minInlinerNum, int lineNumber, Mat *image)
{
	// the points' mask to see if it has been applied to a line already
	std::vector<int> mask(points.size(), 0);
	for (int lineIdx = 0; lineIdx < lineNumber; lineIdx++) {
		// The new set of inliers
		inliers.emplace_back(std::vector<int>());
		// The parameters of the new line
		lines.emplace_back(cv::Mat(3, 1, CV_64F));

		// RANSAC to find the line parameters and the inliers
		this->FitLineRANSAC(
			points,						// The generated 2D points
			mask,						// The point's mask
			inliers.back(),				// Output: the indices of the inliers
			lines.back(),				// Output: the parameters of the found 2D line
			threshold,					// The inlier-outlier threshold
			confidence,					// The confidence required in the results
			iteration_number,			// The number of iterations
			image);						// Optional: the image where we can draw results

		if (inliers.back().size() < minInlinerNum) {
			inliers.resize(inliers.size() - 1);
			lines.resize(lines.size() - 1);
			break;
		}

		for (const auto &inlierIdx : inliers.back())
			mask[inlierIdx] = lineIdx + 1;

	}
}

// Apply RANSAC to fit points to a 2D line
void LineFit::FitLineRANSAC(const vector<Point2d> &points_, const vector<int> &mask_, vector<int> &inliers_, Mat &line_,
	double threshold_, double confidence_, int maximum_iteration_number_, Mat *image_)
{
	// The current number of iterations
	int iterationNumber = 0;

	// The indices of the inliers of the current best model
	vector<int> bestInliers, inliers;
	bestInliers.reserve(points_.size());
	inliers.reserve(points_.size());
	// The parameters of the best line
	Mat bestLine(3, 1, CV_64F);
	// Helpers to draw the line if needed
	Point2d bestPt1, bestPt2;
	// The sample size, i.e., 2 for 2D lines
	constexpr int kSampleSize = 2;
	// The current sample
	std::vector<int> sample(kSampleSize);

	cv::Mat tmp_image;
	size_t maximumIterations = maximum_iteration_number_;

	while (iterationNumber++ < maximumIterations)
	{
		// 1. Select a minimal sample, i.e., in this case, 2 random points.
		for (size_t sampleIdx = 0; sampleIdx < kSampleSize; ++sampleIdx)
		{
			do
			{
				// Generate a random index between [0, pointNumber]
				sample[sampleIdx] =
					round((points_.size() - 1) * static_cast<double>(rand()) / static_cast<double>(RAND_MAX));

				if (mask_[sample[sampleIdx]] != 0)
					continue;

				// If the first point is selected we don't have to check if
				// that particular index had already been selected.
				if (sampleIdx == 0)
					break;

				// If the second point is being generated,
				// it should be checked if the index had been selected beforehand. 
				if (sampleIdx == 1 &&
					sample[0] != sample[1])
					break;
			} while (true);
		}
		// 2. Fit a line to the points.
		const Point2d &p1 = points_[sample[0]]; // First point selected
		const Point2d &p2 = points_[sample[1]]; // Second point select		
		Point2d v = p2 - p1; // Direction of the line
		// cv::norm(v) = sqrt(v.x * v.x + v.y * v.y)
		v = v / cv::norm(v);
		// Rotate v by 90° to get n.
		Point2d n;
		n.x = -v.y;
		n.y = v.x;
		// To get c use a point from the line.
		double a = n.x;
		double b = n.y;
		double c = -(a * p1.x + b * p1.y);


		// 3. Count the number of inliers, i.e., the points closer than the threshold.
		inliers.clear();
		for (size_t pointIdx = 0; pointIdx < points_.size(); ++pointIdx)
		{
			if (mask_[pointIdx] != 0)
				continue;

			const Point2d &point = points_[pointIdx];
			const double distance =
				abs(a * point.x + b * point.y + c);

			if (distance < threshold_)
			{
				inliers.emplace_back(pointIdx);
			}
		}

		// 4. Store the inlier number and the line parameters if it is better than the previous best. 
		if (inliers.size() > bestInliers.size())
		{
			bestInliers.swap(inliers);
			inliers.clear();
			inliers.resize(0);

			bestLine.at<double>(0) = a;
			bestLine.at<double>(1) = b;
			bestLine.at<double>(2) = c;

			// Update the maximum iteration number
			maximumIterations =GetIterationNumber(
				static_cast<double>(bestInliers.size()) / static_cast<double>(points_.size()),
				confidence_,
				kSampleSize);

			//printf("Inlier number = %d\n", bestInliers.size());
			cout << "Inlier number :" << bestInliers.size() << " max iterations :" << maximumIterations << endl;
		}
	}

	inliers_ = bestInliers;
	line_ = bestLine;
}

size_t LineFit::GetIterationNumber(const double &inlierRatio_, const double &confidence_, const size_t &sampleSize_)
{
	double a =
		log(1.0 - confidence_);
	double b =
		log(1.0 - std::pow(inlierRatio_, sampleSize_));

	if (abs(b) < std::numeric_limits<double>::epsilon())
		return std::numeric_limits<size_t>::max();

	return a / b;
}


