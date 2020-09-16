#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <stdio.h>
#include "opencv2/opencv.hpp"
#include <iostream>
#include <sstream>
#include <stdlib.h>


using namespace cv;
using namespace std;
int MAX_KERNEL_LENGTH1 = 15;

Mat frame;
Mat dst;
int thresh = 100;
int max_thresh = 255;
RNG rng(12345);

void thresh_callback(int, void*);


int main(int argc, char** argv)
{
	//Mat frame;
	Mat background;
	Mat object;
	Mat src, src_gray;
	Mat grad;

	int scale = 1;
	int delta = 0;
	int ddepth = CV_16S;

	VideoCapture cap("C:\\Users\\acer\\source\\repos\\opencvTry\\opencvTry\\video.avi"); // open the default camera
	if (!cap.isOpened())  // check if we succeeded
		return -1;

	cap.read(frame);
	Mat acc = Mat::zeros(frame.size(), CV_32FC1); //ขาวดำ

	namedWindow("frame", 1);
	/*
	namedWindow("Running Average");
	namedWindow("Threshold");
	*/
	for (;;)
	{
		Mat gray;
		cap >> frame; // get a new frame from camera //รับภาพมา
		imshow("frame", frame);

		// Get 50% of the new frame and add it to 50% of the accumulator



		cvtColor(frame, frame, COLOR_BGR2GRAY);
		equalizeHist(frame, dst); //เพิ่มความคมชัดอัตโนมัติ

		GaussianBlur(dst, frame, Size(21, 21), 0, 0); // code low-pass
		accumulateWeighted(frame, acc, 0.5); // BG

		// Scale it to 8-bit unsigned
		convertScaleAbs(acc, background);
		/*
		imshow("Original", frame);

		imshow("Weighted Average", background);
		*/

		subtract(frame, background, frame); // เปรียบเทียบ BG frame/frame 

		threshold(frame, frame, 10, 255, THRESH_BINARY); //.

		imshow("Threshold", frame);


		///thresh_callback(0, 0);

		Mat grad_x, grad_y;
		Mat abs_grad_x, abs_grad_y;

		/// Gradient X
		//Scharr( src_gray, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );
		Sobel(dst, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT);
		convertScaleAbs(grad_x, abs_grad_x);

		/// Gradient Y
		//Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
		Sobel(dst, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT);
		convertScaleAbs(grad_y, abs_grad_y);

		/// Total Gradient (approximate)
		addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);


		//imshow("Threshold", grad);

		createTrackbar(" Threshold:", "Source", &thresh, max_thresh, thresh_callback);
		thresh_callback(0, 0);

		if (waitKey(30) >= 0) break;
	}
	// the camera will be deinitialized automatically in VideoCapture destructor
	return 0;
}
void thresh_callback(int, void*)
{
	Mat threshold_output;
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;

	/// Detect edges using Threshold
	threshold(frame, threshold_output, thresh, 255, THRESH_BINARY);
	/// Find contours 
	findContours(threshold_output, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));

	/// Approximate contours to polygons + get bounding rects and circles
	vector<vector<Point> > contours_poly(contours.size());
	vector<Rect> boundRect(contours.size());
	vector<Point2f>center(contours.size());
	vector<float>radius(contours.size());
	vector<Rect>blobs;

	for (int i = 0; i < contours.size(); i++)
	{
		approxPolyDP(Mat(contours[i]), contours_poly[i], 3, true);
		boundRect[i] = boundingRect(Mat(contours_poly[i]));
		if (boundRect[i].width > 25 && boundRect[i].height > 35) {
			blobs.push_back(boundRect[i]);
			minEnclosingCircle(contours_poly[i], center[i], radius[i]);
		}
		
	}


	/// Draw polygonal contour + bonding rects + circles
	Mat drawing = Mat::zeros(threshold_output.size(), CV_8UC3);
	for (int i = 0; i < blobs.size(); i++)
	{
		Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
		//drawContours(drawing, contours_poly, i, color, 1, 8, vector<Vec4i>(), 0, Point());
		rectangle(drawing, blobs[i].tl(), boundRect[i].br(), color, 2, 8, 0);
		//circle(drawing, center[i], (int)radius[i], color, 2, 8, 0);
	}

	/// Show in a window
	//namedWindow("Contours", 1);
	imshow("Contours", drawing);
}