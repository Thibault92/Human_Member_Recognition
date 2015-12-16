/*
 * main.cpp
 *
 *  Created on: 16 nov. 2015
 *      Author: tinyl
 */

#include "libraries.h"
#include "lib_opencv.h"
#include "main_features_detect.hpp"
#include "createKernel.hpp"
#include "templateMatching.hpp"
#include "colorHistogram.hpp"

using namespace std;
using namespace cv;

int main(){

	// Loading image to analyse

	cv::Mat img_original, dst, tpl;

	dst = cv::imread("/home/tinyl/Images/hand.jpg");
	//dst = cv::imread("/home/tinyl/Images/Rituals/DSC08075.JPG");
	//dst = cv::imread("/home/tinyl/Images/groupe3.jpg");
	//dst = cv::imread("/home/tinyl/Images/main3.jpg");
	//dst = cv::imread("/home/tinyl/Images/cochon.jpg");
	//dst = cv::imread("/home/tinyl/Images/test.jpg");
	//dst = cv::imread("/home/tinyl/Images/main2.jpg");
	//dst = cv::imread("/home/tinyl/Images/main1.png");
	//dst = cv::imread("/home/tinyl/Images/main.jpg");


	tpl = cv::imread("/home/tinyl/Images/template.jpg");


// ------------------------------------------   Non loading condition  -----------------------------------------------

	if (dst.empty())
	{
		std::cout << "Cannot load image!" << std::endl;
		return -1;
	}

// -------------------------------------------  Resizing image  ------------------------------------------------------

	Size size(1024,768);
	resize(dst, img_original, size);

	// Showing resized image
	cv::namedWindow("Original", CV_WINDOW_AUTOSIZE);
	cv::imshow("Original",img_original);
	cv::waitKey(0);

// ------------------------------------------- Calculating color Histogram ------------------------------------------

	Mat histImage;
	colorHistogram(img_original, histImage);

// -------------------------------------------  Convert image from RGB domain to YCbCr  ------------------------------

	Mat img_YCbCr = Mat::zeros (img_original.rows, img_original.cols, CV_8UC3);
	cvtColor(img_original,img_YCbCr,CV_RGB2YCrCb);

	cv::namedWindow("Modified", CV_WINDOW_AUTOSIZE);
	cv::imshow("Modified",img_YCbCr);
	cv::waitKey(0);


// ------------------------------------  Thresholding image to detect first main features  --------------------------
	Mat imdetect = Mat::zeros(img_YCbCr.rows, img_YCbCr.cols, CV_8UC3);
	main_features_detect(img_YCbCr, imdetect);

	// Showing result
	cv::namedWindow("Result", CV_WINDOW_AUTOSIZE);
	cv::imshow("Result",imdetect);
	cv::waitKey(0);


// ---------------------------------------  Convert YCbCr image to a binary image  -----------------------------------

	Mat img_back_RGB = Mat::zeros(imdetect.rows, imdetect.cols, CV_8UC3);
	cvtColor(imdetect,img_back_RGB,CV_YCrCb2RGB);

	Mat img_gray = Mat::zeros(imdetect.rows, imdetect.cols, CV_8UC1);
	cvtColor(img_back_RGB,img_gray,CV_RGB2GRAY);

	Mat img_bw= Mat (img_gray.size(),img_gray.type());
	threshold(img_gray, img_bw, 100, 255, THRESH_BINARY);

	cv::namedWindow("Binary Image", CV_WINDOW_AUTOSIZE);
	cv::imshow("Binary Image",img_bw);
	cv::waitKey(0);


// ----------------------------------------------  Morphological Transformations  ------------------------------------------------

	// Creating kernel

	Mat elementStruct = Mat::zeros(3,3,CV_8UC1);
	createKernel(elementStruct, 3, 3);

	Mat imgMorpho = Mat(elementStruct.size(),elementStruct.type());

	// Suppressing noise pixels

	cv::dilate(img_bw,imgMorpho,Mat(),Point(1,1),2,1,1);
	cv::erode(imgMorpho,imgMorpho,Mat(),Point(1,1),1,1,1);
	cv::dilate(imgMorpho,imgMorpho,Mat(),Point(1,1),1,1,1);
	cv::erode(imgMorpho,imgMorpho,Mat(),Point(1,1),2,1,1);

	// Detecting closed contours

	cv::Laplacian(imgMorpho,imgMorpho,CV_8U,3,1,0,BORDER_DEFAULT);
	convertScaleAbs(imgMorpho, imgMorpho);

	cv::dilate(imgMorpho,imgMorpho,Mat(),Point(1,1),2,1,1);
	cv::erode(imgMorpho,imgMorpho,Mat(),Point(1,1),2,1,1);
	cv::dilate(imgMorpho,imgMorpho,Mat(),Point(1,1),1,1,1);

	imwrite("imglaplacien.jpg", imgMorpho);
	cv::namedWindow("Laplacien", CV_WINDOW_AUTOSIZE);
	cv::imshow("Laplacien",imgMorpho);
	cv::waitKey(0);

// ------------------------------------- Creation of bounding-boxes and thresholding them ----------------------------


	RNG rng(12345); // Generating random value used for colors
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	// int area = 2500;


	findContours( imgMorpho, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0) );
	vector<vector<Point> > contours_poly( contours.size() );
	vector<Rect> boundRect( contours.size() );
	vector<Rect> boundRectFiltre( contours.size() );

	for( size_t i = 0; i < contours.size(); i++ )
	{
		approxPolyDP( Mat(contours[i]), contours_poly[i], 3, true );
		boundRect[i] = boundingRect( Mat(contours_poly[i]) );
		//cout << "Top-left" << boundRect[i].tl() << endl;
		//cout << "Bottom-right" << boundRect[i].br() << endl;
	}

	Mat drawing = Mat::zeros( imgMorpho.size(), CV_8UC3 );

	int boolean = 1;

	for( size_t i = 0; i< contours.size(); i++ ){
		boolean = 1;

		int x1 = boundRect[i].tl().x;
		int y1 = boundRect[i].tl().y;
		int x2 = boundRect[i].br().x;
		int y2 = boundRect[i].br().y;

		int aire = (x2 - x1)*(y2 - y1);

		//cout << aire << endl;


		if (aire > 500)
		{
			for (size_t j=0 ; j < contours.size() ; j++)
			{

				int tempx1 = boundRect[j].tl().x;
				int tempy1 = boundRect[j].tl().y;
				int tempx2 = boundRect[j].br().x;
				int tempy2 = boundRect[j].br().y;

				if (x1 > tempx1 && x2 < tempx2 && y1 > tempy1 && y2 < tempy2 )
				{
					boolean = 0;

				}
			}

			if (boolean == 1)
			{
				Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
				drawContours( drawing, contours_poly, (int)i, color, 1, 8, vector<Vec4i>(), 0, Point() );
				rectangle( drawing, boundRect[i].tl(), boundRect[i].br(), color, 2, 8, 0 );

			}
		}
	}

//--------------------------------------------------- Creating template mask  -----------------------------------------

	Mat tpl_YCbCr, tplDetect, tpl_back_RGB, tpl_gray;
	tpl_YCbCr = Mat::zeros (tpl.rows, tpl.cols, CV_8UC3);
	cvtColor(tpl,tpl_YCbCr,CV_RGB2YCrCb);

	tplDetect = Mat::zeros(tpl_YCbCr.rows, tpl_YCbCr.cols, CV_8UC3);
    main_features_detect(tpl_YCbCr, tplDetect);

    tpl_back_RGB = Mat::zeros(tplDetect.rows, tplDetect.cols, CV_8UC3);
    cvtColor(tplDetect,tpl_back_RGB,CV_YCrCb2RGB);
    tpl_gray = Mat::zeros(tplDetect.rows, tplDetect.cols, CV_8UC1);
    cvtColor(tpl_back_RGB,tpl_gray,CV_RGB2GRAY);

	//imwrite("imgrogner.jpg", drawing);
	namedWindow( "Contours", WINDOW_AUTOSIZE );
	imshow( "Contours", drawing );
	waitKey(0);

// ------------------------------------------------ Template Matching --------------------------------------------------

	templateMatching(img_gray, tpl_gray);
	imshow( "Matching", img_gray );
	waitKey(0);

	/*	Mat subImage = img_original(boundRect[i]);
		//imwrite("imgsub.jpg", subImage);
		cv::namedWindow("sub", CV_WINDOW_AUTOSIZE);
		imshow("sub", subImage);*/


}

