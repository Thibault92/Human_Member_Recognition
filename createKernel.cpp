/*
 * createKernel.cpp
 *
 *  Created on: 10 déc. 2015
 *      Author: tinyl
 */

#include "lib_opencv.h"
#include "libraries.h"

using namespace cv;
using namespace std;

Mat createKernel(Mat kernel, int height, int width){

	for (int i = 0; i < height ; i ++)
		    {
		    	for (int j = 0 ; j < width ; j++)
		    	{
		    		kernel.at < Vec3b > (Point(i, j)) = 255;
		    		//kernel.at < Vec3b > (Point(i, j))[1] = 255;
		    		//kernel.at < Vec3b > (Point(i, j))[2] = 255;
		    	}
		    }

	return kernel;
}


