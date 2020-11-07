/*
1. Read LiDAR point cloud
2. Find most dominant plane on the point cloud
3. Implement LO-RANSAC for plane fitting
4. Local optimisation methods - LSQ fitting for inliers, iterated LSQ, inner RANSAC
5. Use colored points to distinguish fitted planes
*/

#include "OptimisedRANSAC.h"

loRANSAC::loRANSAC(char* fileName, int iterations) {
	this->fileName = fileName;
	this->iterations = iterations;
}

void loRANSAC::LocallyOptimisedRANSAC(int option) {
	/*load data to get point data*/
	/*1: x coord. 2. y coord, 3. z coord, 4. R value, 5. G value, 6. B value*/
	vector<allPoints> points;
	this->LoadData(points);
	cout << "Number of points: " << points.size() << endl;
	/*Fit the plane*/
	vector<int> inliers;
	double threshold = 0.009;
	double loThreshold = 2.;
	double confidence = 0.9999;
	Mat plane;
	this->FitLoRANSAC(points, this->iterations, threshold, loThreshold, confidence, inliers, plane, option);
	/*For the best set of inliers found, color them and save them in an xyz file*/
	cout << inliers.size() << endl;
	for (int i = 0; i < inliers.size(); i++) {
		points[inliers[i]].green = 255;
	}

	// Calculate the error or the Least Squares Fitting line
	double a = plane.at<double>(0);
	double b = plane.at<double>(1);
	double c = plane.at<double>(2);
	double d = plane.at<double>(3);
	cout << a << " " << b << " " << c << " " << d << endl;
	double averageError = 0.0;
	for(int inlierIdx = 0; inlierIdx < inliers.size(); inlierIdx++){
		Point3d p = points[inliers[inlierIdx]].point;
		double distance = abs(a * p.x + b * p.y + c * p.z + d);
		averageError += distance;
	}
	averageError /= inliers.size();

	cout << "Average RANSAC error for plane is : " << averageError << endl;
	/*write the inlier values to xyz file*/
	this->WriteDataPoints(points);
}

void loRANSAC::WriteDataPoints(vector<allPoints>& points) {
	string fileName = "PlaneFitting.xyz";
	ofstream dataFile;
	dataFile.open(fileName);
	cout << "Writing into xyz file" << endl;
	for (int i = 0; i < points.size(); i++) {
		dataFile << points[i].point.x << " " << points[i].point.y << " " << points[i].point.z << " " << 
				  points[i].red << " " << points[i].green << " " << points[i].blue << " " << endl;
	}
	dataFile.close();
}



void loRANSAC::LoadData(vector<allPoints>& points) {
	ifstream datafile(fileName);
	string line;
	int lineCounter = 0;
	if (datafile.is_open()) {
		while (!datafile.eof()) {
			Point3d point;
			vector<string> coords;
			string word = " ";
			getline(datafile, line);
			for (char l : line) {
				if (l == ' ') {
					//cout << word << endl;
					coords.push_back(word);
					word = "";
				}
				else {
					word = word + l;
				}
			}
			//cout << word << endl;
			coords.push_back(word);
			size_t sz;
			point.x = stod(coords[0], &sz);
			point.y = stod(coords[1], &sz);
			point.z = stod(coords[2], &sz);
			int r = stoi(coords[3]);
			/*RGB in xyz file, BGR in OpenCV, hence order is flipped*/
			allPoints aP = { point, stoi(coords[5], &sz) , stoi(coords[4], &sz) , stoi(coords[3], &sz) };
			points.emplace_back(aP);
		}
	}
	else {
		cerr << "Cannot open file" << endl;
		exit(-1);
	}
}


void loRANSAC::FitPlaneLSQ(const vector<allPoints>& points_, vector<int>& inliers_, Mat& bestPlane) {

	vector<Point3d> normalizedPoints;
	normalizedPoints.reserve(inliers_.size());

	// Calculating the mass point of the points
	Point3d masspoint(0, 0, 0);

	/*for all the inliers, centering the mass of the points*/
	for (const auto &inlierIdx : inliers_)
	{
		masspoint += points_[inlierIdx].point;
		normalizedPoints.emplace_back(points_[inlierIdx].point);
	}
	masspoint = masspoint * (1.0 / inliers_.size());

	// Move the point cloud to have the origin in their mass point
	for (auto &point : normalizedPoints)
		point -= masspoint;

	// Calculating the average distance from the origin
	double averageDistance = 0.0;
	for (auto &point : normalizedPoints)
	{
		averageDistance += cv::norm(point);
		// norm(point) = sqrt(point.x * point.x + point.y * point.y)
	}

	averageDistance /= normalizedPoints.size();
	const double ratio = sqrt(2) / averageDistance;

	// Making the average distance to be sqrt(2)
	for (auto &point : normalizedPoints)
		point *= ratio;

	// Now, we should solve the equation.
	cv::Mat A(normalizedPoints.size(), 2, CV_64F);

	// Building the coefficient matrix
	for (size_t pointIdx = 0; pointIdx < normalizedPoints.size(); ++pointIdx)
	{
		const size_t &rowIdx = pointIdx;

		A.at<double>(rowIdx, 0) = normalizedPoints[pointIdx].x;
		A.at<double>(rowIdx, 1) = normalizedPoints[pointIdx].y;
	}

	cv::Mat evals, evecs;
	cv::eigen(A.t() * A, evals, evecs);

	const cv::Mat &normal = evecs.row(1);

	const double &a = normal.at<double>(0),
				 &b = normal.at<double>(1),
				 &c = normal.at<double>(2);
	const double d = -(a * masspoint.x + b * masspoint.y + c * masspoint.z);

	bestPlane.at<double>(0) = a;
	bestPlane.at<double>(1) = b;
	bestPlane.at<double>(2) = c;
	bestPlane.at<double>(3) = d;
	//cout << a << " " << b << " " << c << " " << d << endl;
}

void loRANSAC::FitLoRANSAC(const vector<allPoints>& points_, int maximum_iteration_number_, double threshold_, 
							double loThreshold, double confidence_, vector<int>& inliers_, Mat& plane_, int option) {
	// The current number of iterations
	int iterationNumber = 0;
	//cout << threshold << endl;
	// The indices of the inliers of the current best model
	vector<int> bestInliers, inliers;
	bestInliers.reserve(points_.size());
	inliers.reserve(points_.size());
	// The parameters of the best plane
	Mat bestPlane(4, 1, CV_64F);
	//Local optimisation parameters
	vector<int> bestLoInliers, loInliers;
	Mat bestLoPlane(4, 1, CV_64F);
	bestLoInliers.reserve(bestInliers.size());
	loInliers.reserve(bestInliers.size());

	// The sample size, i.e., 3 for plane
	constexpr int kSampleSize = 3;
	// The current sample
	std::vector<int> sample(kSampleSize);

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

				// If the first point is selected we don't have to check if
				// that particular index had already been selected.
				if (sampleIdx == 0)
					break;

				// If the second point is being generated,
				// it should be checked if the index had been selected beforehand. 
				if (sampleIdx == 1 &&
					(sample[0] != sample[1]))
					break;
				if (sampleIdx == 2 &&
					(sample[0] != sample[2]) && (sample[1] != sample[2]))
					break;
			} while (true);
		}
		// 2. Fit a line to the points.
		const Point3d &p1 = points_[sample[0]].point; // First point selected
		const Point3d &p2 = points_[sample[1]].point; // Second point select	
		const Point3d &p3 = points_[sample[2]].point; // Third point selected
		Point3d v1 = p2 - p1; // Direction of the line 1
		// cv::norm(v) = sqrt(v.x * v.x + v.y * v.y)
		v1 = v1 / cv::norm(v1);
		Point3d v2 = p3 - p1; //Direction of line 2
		v2 = v2 / cv::norm(v2);
		// Calculate parameters of plane by calculating normal to the lines
		Point3d n;
		n.x = (v1.y * v2.z) - (v2.y * v1.z);
		n.y = -(v1.x * v2.z) + (v2.x * v1.z);
		n.z = (v1.x * v2.y) - (v2.x * v1.y);
		// To get c use a point from the line.
		double a = n.x;
		double b = n.y;
		double c = n.z;
		double d = -(a * p1.x + b * p1.y + c * p1.z);
		//cout << a << " " << b << " " << c << " " << d << endl;

		// 3. Count the number of inliers, i.e., the points closer than the threshold.
		inliers.clear();
		for (size_t pointIdx = 0; pointIdx < points_.size(); ++pointIdx)
		{

			const Point3d &point = points_[pointIdx].point;
			const double distance = abs(a * point.x + b * point.y + c * point.z + d);
			//cout << distance << endl;
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
			int loIterationNum = 0;
			size_t maxLoIterationNum = maximum_iteration_number_;
			switch (option) {
			case 1:
				/*find new model parameters for the best inliers based on least square fitting*/
				this->FitPlaneLSQ(points_, bestInliers, bestPlane);
				a = bestPlane.at<double>(0);
				b = bestPlane.at<double>(1);
				c = bestPlane.at<double>(2);
				d = bestPlane.at<double>(3);
				// Update the maximum iteration number
				maximumIterations = GetIterationNumber(
					static_cast<double>(bestInliers.size()) / static_cast<double>(points_.size()),
					confidence_,
					kSampleSize);

				//printf("Inlier number = %d\n", bestInliers.size());
				cout << "Inlier number :" << bestInliers.size() << " max iterations :" << maximumIterations << endl;
				break;
			case 2:
				/*find new model parameters for the best inliers based on least square fitting*/
				this->FitPlaneLSQ(points_, bestInliers, bestPlane);
				a = bestPlane.at<double>(0);
				b = bestPlane.at<double>(1);
				c = bestPlane.at<double>(2);
				d = bestPlane.at<double>(3);
				//cout << a << " " << b << " " << c << " " << d << endl;
				loInliers.clear();
				for (size_t pointIdx = 0; pointIdx < bestInliers.size(); ++pointIdx)
				{
					const Point3d &point = points_[bestInliers[pointIdx]].point;
					//cout << bestInliers[pointIdx] << " " << point.x << " " << point.y << " " << point.z << endl;
					const double distance = abs(a * point.x + b * point.y + c * point.z + d);
					//cout << distance << endl;
					if (distance < loThreshold)
					{
						loInliers.emplace_back(bestInliers[pointIdx]);
					}
				}
				while (loIterationNum++ < maxLoIterationNum) {			
					// 4. Store the inlier number and the line parameters if it is better than the previous best. 
					if (loInliers.size() > bestLoInliers.size())
					{
						bestLoInliers.swap(loInliers);
						loInliers.clear();
						loInliers.resize(0);

						bestLoPlane.at<double>(0) = a;
						bestLoPlane.at<double>(1) = b;
						bestLoPlane.at<double>(2) = c;
						bestLoPlane.at<double>(3) = d;

						// Update the maximum iteration number
						maximumIterations = GetIterationNumber(
							static_cast<double>(bestLoInliers.size()) / static_cast<double>(bestInliers.size()),
							confidence_,
							kSampleSize);

						//printf("Inlier number = %d\n", bestInliers.size());
						cout << "local optimised inlier number :" << bestLoInliers.size() << " max local optimised iterations :" << maximumIterations << endl;
					}
				}
				bestPlane.at<double>(0) = bestLoPlane.at<double>(0);
				bestPlane.at<double>(1) = bestLoPlane.at<double>(1);
				bestPlane.at<double>(2) = bestLoPlane.at<double>(2);
				bestPlane.at<double>(3) = bestLoPlane.at<double>(3);
				// Update the maximum iteration number
				maximumIterations = GetIterationNumber(
					static_cast<double>(bestInliers.size()) / static_cast<double>(points_.size()),
					confidence_,
					kSampleSize);

				//printf("Inlier number = %d\n", bestInliers.size());
				cout << "Inlier number :" << bestInliers.size() << " max iterations :" << maximumIterations << endl;
				bestInliers = bestLoInliers;
				break;
			case 3:
				bestPlane.at<double>(0) = a;
				bestPlane.at<double>(1) = b;
				bestPlane.at<double>(2) = c;
				bestPlane.at<double>(3) = d;
				//cout << a << " " << b << " " << c << " " << d << endl;
				// Update the maximum iteration number
				maximumIterations = GetIterationNumber(
					static_cast<double>(bestInliers.size()) / static_cast<double>(points_.size()),
					confidence_,
					kSampleSize);

				//printf("Inlier number = %d\n", bestInliers.size());
				cout << "Inlier number :" << bestInliers.size() << " max iterations :" << maximumIterations << endl;
				break;
			default:
				cout << "Enter valid local optimisation option" << endl;
				break;
			}
		}
	}

	inliers_ = bestInliers;
	plane_ = bestPlane;
	cout << inliers_.size() << " " << plane_.size() << endl;
}

size_t loRANSAC::GetIterationNumber(const double &inlierRatio_, const double &confidence_, const size_t &sampleSize_)
{
	double a =
		log(1.0 - confidence_);
	double b =
		log(1.0 - std::pow(inlierRatio_, sampleSize_));

	if (abs(b) < std::numeric_limits<double>::epsilon())
		return std::numeric_limits<size_t>::max();

	return a / b;
}
