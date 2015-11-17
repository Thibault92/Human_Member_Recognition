/*
 * main.cpp
 *
 *  Created on: 16 nov. 2015
 *      Author: tinyl
 */


#include "libraries.h"
#include "lib_opencv.h"
#include "main_features_detect.hpp"

using namespace std;
using namespace cv;

int main(){

	cv::Mat img_original, dst;
	//img_original = cv::imread("/home/tinyl/Images/main3.jpg");
	//img_original = cv::imread("/home/tinyl/Images/cochon.jpg");
	//img_original = cv::imread("/home/tinyl/Images/test.jpg");
	img_original = cv::imread("/home/tinyl/Images/main2.jpg");
	//img_original = cv::imread("/home/tinyl/Images/main1.png");
	//img_original = cv::imread("/home/tinyl/Images/main.jpg");

	//Condition de non lecture
		  if (img_original.empty())
		    {
		      std::cout << "Cannot load image!" << std::endl;
		      return -1;
		    }

	    cv::namedWindow("Original", CV_WINDOW_AUTOSIZE);
	    cv::imshow("Original",img_original);
	    cv::waitKey(0);

	    Mat img_YCbCr;
	    img_YCbCr = Mat::zeros (img_original.rows, img_original.cols, CV_8UC3);
	    cvtColor(img_original,img_YCbCr,CV_RGB2YCrCb);

	    cv::namedWindow("Modified", CV_WINDOW_AUTOSIZE);
	    cv::imshow("Modified",img_YCbCr);
	    cv::waitKey(0);

	    cout << "Valeur YCrCb:" << img_YCbCr.at<Vec3b>(Point(35,90));
	    cout << endl;
	    cout << "Valeur RGB:" << img_original.at<Vec3b>(Point(35,90));
	    cout << endl;
	    cout << float(img_YCbCr.at<Vec3b>(Point(35,90))[1])/float(img_YCbCr.at<Vec3b>(Point(35,90))[2]);
	    cout << endl;


	    Mat imdetect;
	    imdetect = Mat::zeros(img_YCbCr.rows, img_YCbCr.cols, CV_8UC3);

	    main_features_detect(img_YCbCr, imdetect);

	    /* for(int i = 0; i < img_YCbCr.rows; i++){
	    	for(int j=0; j < img_YCbCr.cols; j++){

	    		if (float(img_YCbCr.at<Vec3b>(Point(j,i))[1])/float(img_YCbCr.at<Vec3b>(Point(j,i))[2]) > 0.65
	    				&& float(img_YCbCr.at<Vec3b>(Point(j,i))[1])/float(img_YCbCr.at<Vec3b>(Point(j,i))[2]) < 0.8){
	    			imdetect.at<Vec3b>(Point(j,i))[0] = 255; //bleu
	    			imdetect.at<Vec3b>(Point(j,i))[1] = 255;	//vert
	    			imdetect.at<Vec3b>(Point(j,i))[2] = 255;	//rouge

	    		}

	    	}
	    } */


	  /*  Mat imdetect;
	    imdetect = Mat::zeros(img_original.rows, img_original.cols, CV_8UC3);

	    Mat R;
	    R = Mat::zeros(img_original.rows, img_original.cols, CV_8UC3);
	    Mat G;
	    G = Mat::zeros(img_original.rows, img_original.cols, CV_8UC3);
	    Mat B;
	    B = Mat::zeros(img_original.rows, img_original.cols, CV_8UC3);


	    for(int i = 0; i < img_original.rows; i++){
	    	for(int j=0; j < img_original.cols; j++){

	    		B.at<Vec3b>(Point(j,i)) = img_original.at<Vec3b>(Point(j,i))[0];
	    		G.at<Vec3b>(Point(j,i))[1] = img_original.at<Vec3b>(Point(j,i))[1];
	    		R.at<Vec3b>(Point(j,i))[2] = img_original.at<Vec3b>(Point(j,i))[2];


	    	 }
	    }*/

	    cv::namedWindow("Result", CV_WINDOW_AUTOSIZE);
	    cv::imshow("Result",imdetect);
	    cv::waitKey(0);


	    /// Separate the image in 3 places ( B, G and R )
	      vector<Mat> bgr_planes;
	      split( img_YCbCr, bgr_planes );

	      /// Establish the number of bins
	      int histSize = 256;

	      /// Set the ranges ( for B,G,R) )
	      float range[] = { 0, 256 } ;
	      const float* histRange = { range };

	      bool uniform = true; bool accumulate = false;

	      cv::Mat b_hist, g_hist, r_hist;

	      /// Compute the histograms:
	      calcHist( &bgr_planes[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate );
	      calcHist( &bgr_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate );
	      calcHist( &bgr_planes[2], 1, 0, Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate );

	      // Draw the histograms for B, G and R
	      int hist_w = 512; int hist_h = 400;
	      int bin_w = cvRound( (double) hist_w/histSize );

	      Mat histImage( hist_h, hist_w, CV_8UC3, Scalar( 0,0,0) );

	      /// Normalize the result to [ 0, histImage.rows ]
	      normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
	      normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
	      normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );

	      /// Draw for each channel
	      for( int i = 1; i < histSize; i++ )
	      {
	          line( histImage, Point( bin_w*(i-1), hist_h - cvRound(b_hist.at<float>(i-1)) ) ,
	                           Point( bin_w*(i), hist_h - cvRound(b_hist.at<float>(i)) ),
	                           Scalar( 255, 0, 0), 2, 8, 0  );
	          line( histImage, Point( bin_w*(i-1), hist_h - cvRound(g_hist.at<float>(i-1)) ) ,
	                           Point( bin_w*(i), hist_h - cvRound(g_hist.at<float>(i)) ),
	                           Scalar( 0, 255, 0), 2, 8, 0  );
	          line( histImage, Point( bin_w*(i-1), hist_h - cvRound(r_hist.at<float>(i-1)) ) ,
	                           Point( bin_w*(i), hist_h - cvRound(r_hist.at<float>(i)) ),
	                           Scalar( 0, 0, 255), 2, 8, 0  );
	      }

	      /// Display
	      namedWindow("calcHist Demo", CV_WINDOW_AUTOSIZE );
	      imshow("calcHist Demo", histImage );

	      waitKey(0);

	      return 0;



}

