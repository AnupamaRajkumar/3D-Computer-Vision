#pragma once
#ifndef __SFM__
#define __SFM__

#include <fstream>
#include <iostream>
#include <vector>
#include <chrono>  
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>


class SFM {
public:
	SFM(cv::Mat &image1, cv::Mat &image2);
	void SFMOperation();

private:
	cv::Mat image1, image2;
	// Detecting point correspondences in two images
	void FeatureMatching(cv::Mat& image1,
						 cv::Mat& image2,
						std::vector<cv::Point2d> &source_points,
						std::vector<cv::Point2d> &destimation_points);


	// A function estimating the fundamental matrix from point correspondences
	// by RANSAC.
	void ransacFundamentalMatrix(
		const std::vector<cv::Point2d> &input_source_points_,					// Points in the source image
		const std::vector<cv::Point2d> &input_destination_points_,				// Points in the destination image
		const std::vector<cv::Point2d> &normalized_input_src_points_,			// Normalized points in the source image
		const std::vector<cv::Point2d> &normalized_input_destination_points_,	// Normalized points in the destination image
		const cv::Mat &T1_,								// Normalizing transformation in the source image
		const cv::Mat &T2_,								// Normalizing transformation in the destination image
		cv::Mat &fundamental_matrix_,					// The estimated fundamental matrix
		std::vector<size_t> &inliers_,					// The inliers of the fundamental matrix
		double confidence_,								// The required confidence of RANSAC
		double threshold_);								// The inlier-outlier threshold

	// A function estimating the fundamental matrix from point correspondences
	// by least-squares fitting.
	void getFundamentalMatrixLSQ(
		const std::vector<cv::Point2d> &source_points_,		 // Points in the source image
		const std::vector<cv::Point2d> &destination_points_, // Points in the destination image
		cv::Mat &fundamental_matrix_);						 // The estimated fundamental matrix


	// A function decomposing the essential matrix to the projection matrices
	// of the two cameras.
	void getProjectionMatrices(
		const cv::Mat &essential_matrix_,					// The parameters of the essential matrix
		const cv::Mat &K1_,									// The intrinsic camera parameters of the source image
		const cv::Mat &K2_,									// The intrinsic camera parameters of the destination image
		const cv::Mat &src_point_,							// A point in the source image
		const cv::Mat &dst_point_,							// A point in the destination image
		cv::Mat &projection_1_,								// The projection matrix of the source image
		cv::Mat &projection_2_);							// The projection matrix of the destination image

	// A function estimating the 3D point coordinates from a point correspondences
	// from the projection matrices of the two observing cameras.
	void linearTriangulation(
		const cv::Mat &projection_1_,						// The projection matrix of the source image
		const cv::Mat &projection_2_,						// The projection matrix of the destination image
		const cv::Mat &src_point_,							// A point in the source image
		const cv::Mat &dst_point_,							// A point in the destination image
		cv::Mat &point3d_);									// The estimated 3D coordinates

	// Normalizing the point coordinates for the fundamental matrix estimation
	void normalizePoints(
		const std::vector<cv::Point2d> &input_source_points_,		// Points in the source image
		const std::vector<cv::Point2d> &input_destination_points_,	// Points in the destination image
		std::vector<cv::Point2d> &output_source_points_,			// Normalized points in the source image
		std::vector<cv::Point2d> &output_destination_points_,		// Normalized points in the destination image
		cv::Mat &T1_,												// Normalizing transformation in the source image
		cv::Mat &T2_);												// Normalizing transformation in the destination image

	// Return the iteration number of RANSAC given the inlier ratio and
	// a user-defined confidence value.
	int getIterationNumber(int point_number_,			// The number of points
		int inlier_number_,								// The number of inliers
		int sample_size_,								// The sample size
		double confidence_);							// The required confidence

	// Printing the time to the console
	void printTimes(
		const std::chrono::time_point<std::chrono::system_clock> &start,	// The starting time
		const std::chrono::time_point<std::chrono::system_clock> &end,		// The current time
		const std::string &message);										// The message to be written

	// Visualize the effect of the point normalization
	cv::Mat checkEffectOfNormalization(const std::vector<cv::Point2d> &source_points_,  // Points in the first image 
		const std::vector<cv::Point2d> &destination_points_,							// Points in the second image
		const std::vector<cv::Point2d> &normalized_source_points_,						// Normalized points in the first image 
		const std::vector<cv::Point2d> &normalized_destination_points_,					// Normalized points in the second image
		const cv::Mat &T1_,						// Normalizing transforcv::Mation in the first image
		const cv::Mat &T2_,						// Normalizing transforcv::Mation in the second image
		const std::vector<size_t> &inliers_);	// The inliers of the fundamental matrix
};
#endif // !__SFM__
