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
	//dst = cv::imread("/home/tinyl/Images/Rituals/DSC08075.JPG");
	//dst = cv::imread("/home/tinyl/Images/groupe3.jpg");
	//dst = cv::imread("/home/tinyl/Images/main3.jpg");
	//dst = cv::imread("/home/tinyl/Images/cochon.jpg");
	//dst = cv::imread("/home/tinyl/Images/test.jpg");
	//dst = cv::imread("/home/tinyl/Images/main2.jpg");
	dst = cv::imread("/home/tinyl/Images/main1.png");
	//dst = cv::imread("/home/tinyl/Images/main.jpg");

	//Condition de non lecture
		  if (dst.empty())
		    {
		      std::cout << "Cannot load image!" << std::endl;
		      return -1;
		    }

		Size size(1024,768);
		resize(dst, img_original, size);
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


	    cv::namedWindow("Result", CV_WINDOW_AUTOSIZE);
	    cv::imshow("Result",imdetect);
	    cv::waitKey(0);

	    Mat img_back_RGB;
	    img_back_RGB = Mat::zeros(imdetect.rows, imdetect.cols, CV_8UC3);
	    cvtColor(imdetect,img_back_RGB,CV_YCrCb2RGB);

	    /*cv::namedWindow("back_RGB", CV_WINDOW_AUTOSIZE);
	    cv::imshow("back_RGB",img_back_RGB);
	    cv::waitKey(0);
*/

	    Mat img_gray;
	    img_gray = Mat::zeros(imdetect.rows, imdetect.cols, CV_8UC1);
	    cvtColor(img_back_RGB,img_gray,CV_RGB2GRAY);

	    /*cv::namedWindow("gray", CV_WINDOW_AUTOSIZE);
	    cv::imshow("gray",img_gray);
	    cv::waitKey(0);
	    */

	    Mat img_bw; // = img_gray > 128;
	    img_bw = Mat (img_gray.size(),img_gray.type());
	    threshold(img_gray, img_bw, 100, 255, THRESH_BINARY);

	    imwrite("image_bw.jpg", img_bw);
	    cv::namedWindow("Binary Image", CV_WINDOW_AUTOSIZE);
	    cv::imshow("Binary Image",img_bw);
	    cv::waitKey(0);


	    //Creation de l'element structurant
	    Mat elementStruct;
	    elementStruct = Mat::zeros(3,3,CV_8UC3);


	    for (int i = 0; i < 3 ; i ++)
	    {
	    	for (int j = 0 ; j < 3 ; j++)
	    	{
	    		elementStruct.at < Vec3b > (Point(i, j))[0] = 255;
	    		elementStruct.at < Vec3b > (Point(i, j))[1] = 255;
	    		elementStruct.at < Vec3b > (Point(i, j))[2] = 255;
	    	}
	    }
	    Mat elt_gris;
	    elt_gris = Mat::zeros(elementStruct.rows, elementStruct.cols, CV_8UC1);
	    cvtColor(elementStruct,elt_gris,CV_RGB2GRAY);

	    Mat elt_bw; // = img_gray > 128;
	    elt_bw = Mat(elt_gris.size(),elt_gris.type());
	    threshold(elt_gris, elt_bw, 100, 255, THRESH_BINARY);
	    imwrite("element1.jpg", elt_bw);


	    Mat imgMorpho;
	    imgMorpho = Mat(elt_bw.size(),elt_bw.type());

	    cv::dilate(img_bw,imgMorpho,Mat(),Point(1,1),2,1,1);
	    cv::erode(imgMorpho,imgMorpho,Mat(),Point(1,1),1,1,1);
	    cv::dilate(imgMorpho,imgMorpho,Mat(),Point(1,1),1,1,1);
	    cv::erode(imgMorpho,imgMorpho,Mat(),Point(1,1),2,1,1);

	    imwrite("img_erode.jpg", imgMorpho);

	    /*int scale = 1;
	    int delta = 0;
	    int ddepth = CV_16S;
	    Mat imgx;
	    Mat imgy;
	    cv::Sobel(nouvelleImage,imgx,ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
	    cv::Sobel(nouvelleImage,imgy,ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
	    convertScaleAbs( imgx, imgy );
	    imwrite("imgx.jpg",imgx);
	    imwrite("imgSobel.jpg", imgy);*/


	    cv::Laplacian(imgMorpho,imgMorpho,CV_8U,3,1,0,BORDER_DEFAULT);
	    convertScaleAbs( imgMorpho, imgMorpho );


	    cv::dilate(imgMorpho,imgMorpho,Mat(),Point(1,1),2,1,1);
	    cv::erode(imgMorpho,imgMorpho,Mat(),Point(1,1),2,1,1);
	    cv::dilate(imgMorpho,imgMorpho,Mat(),Point(1,1),1,1,1);

	    imwrite("imglaplacien.jpg", imgMorpho);
	    cv::namedWindow("Laplacien", CV_WINDOW_AUTOSIZE);
	    cv::imshow("Laplacien",imgMorpho);
	    cv::waitKey(0);


	    RNG rng(12345);
	    vector<vector<Point> > contours;
	    vector<Vec4i> hierarchy;
	    int area = 2500;


	    findContours( imgMorpho, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0) );
	    vector<vector<Point> > contours_poly( contours.size() );
	    vector<Rect> boundRect( contours.size() );
	    vector<Rect> boundRectFiltre( contours.size() );
	    vector<Point2f>center( contours.size() );
	    vector<float>radius( contours.size() );
	    for( size_t i = 0; i < contours.size(); i++ )
	       {
	    	approxPolyDP( Mat(contours[i]), contours_poly[i], 3, true );
	         boundRect[i] = boundingRect( Mat(contours_poly[i]) );
	         cout << "Top-left" << boundRect[i].tl() << endl;
	         cout << "Bottom-right" << boundRect[i].br() << endl;


	        if(boundRect[i].width * boundRect[i].height > area){
	        	boundRectFiltre[i] = boundRect[i];
	        }
	         //minEnclosingCircle( contours_poly[i], center[i], radius[i] );
	       }

	    Mat drawing = Mat::zeros( imgMorpho.size(), CV_8UC3 );
	    for( size_t i = 0; i< contours.size(); i++ )
	       {

	         Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
	         drawContours( drawing, contours_poly, (int)i, color, 1, 8, vector<Vec4i>(), 0, Point() );
	         rectangle( drawing, boundRectFiltre[i].tl(), boundRectFiltre[i].br(), color, 2, 8, 0 );
	         //circle( drawing, center[i], (int)radius[i], color, 2, 8, 0 );
	       }
	    namedWindow( "Contours", WINDOW_AUTOSIZE );
	    imshow( "Contours", drawing );
	    waitKey(0);


/*	    /// Separate the image in 3 places ( B, G and R )
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

*/

}

