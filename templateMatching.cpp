/*
 * templateMatching.cpp
 *
 *  Created on: 16 déc. 2015
 *      Author: tinyl
 */


#include "lib_opencv.h"
#include "libraries.h"

using namespace cv;
using namespace std;

Mat templateMatching(Mat img, Mat tpl){

		Mat imgResult;
		int match_method = CV_TM_SQDIFF_NORMED;
		int res_cols = img.cols - tpl.cols + 1;
		int res_rows = img.rows - tpl.rows + 1;

		imgResult.create( res_cols, res_rows, CV_32FC1 );

		/// Do the Matching and Normalize
		cv::matchTemplate( img, tpl, imgResult, match_method );
		normalize( imgResult, imgResult, 0, 1, NORM_MINMAX, -1, Mat() );

		/// Localizing the best match with minMaxLoc
		double minVal; double maxVal; Point minLoc; Point maxLoc;
		Point matchLoc;

		minMaxLoc( imgResult, &minVal, &maxVal, &minLoc, &maxLoc, Mat() );

		matchLoc = minLoc;

		cout << matchLoc << endl;

		/// Show me what you got
		rectangle( img, matchLoc, Point( matchLoc.x + tpl.cols , matchLoc.y + tpl.rows ), Scalar(0,0,255), 4, 8, 0 );

		/*//imwrite( "result.jpg", img_gray );
		imshow( "Matching", img );
		waitKey(0);*/

		return img;


}

