/*
 * main_features_detect.cpp
 *
 *  Created on: 16 nov. 2015
 *      Author: tinyl
 */

#include "lib_opencv.h"
#include "libraries.h"

using namespace cv;
using namespace std;

Mat main_features_detect(Mat img, Mat img_result){

	img_result = Mat::zeros(img.rows, img.cols, CV_8UC3);

	for(int i = 0; i < img.rows; i++){
		    	for(int j=0; j < img.cols; j++){

		    		if (float(img.at<Vec3b>(Point(j,i))[1])/float(img.at<Vec3b>(Point(j,i))[2]) > 0.60
		    				&& float(img.at<Vec3b>(Point(j,i))[1])/float(img.at<Vec3b>(Point(j,i))[2]) < 0.9){
		    			img_result.at<Vec3b>(Point(j,i))[0] = 255; //bleu
		    			img_result.at<Vec3b>(Point(j,i))[1] = 255;	//vert
		    			img_result.at<Vec3b>(Point(j,i))[2] = 255;	//rouge

		    		}

		    	}
		    }
	return img_result;

}


