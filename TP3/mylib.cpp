#include "mylib.h"
//---------------------noirBlanc-----------------------

Mat noirBlanc(Mat frame)
{ 
	Mat im_gray_out;
	
	if (frame.empty())
	exit(0);
	
	cvtColor(frame,im_gray_out,COLOR_RGB2GRAY);	
	return im_gray_out;
}



//---------------------get_frame-----------------------

Mat get_frame(Mat frame)
{
	if (frame.empty())
	exit(0);
	
	return frame;
}

//---------------------seuillage------------------------

Mat seuillage(Mat frame)
{ 
	float nr;//ratio de rouge
	uchar r,v,b;//niveau de rouge, vert et bleu
	Mat frame_out;
	frame_out.create(frame.rows,frame.cols,CV_8UC3);

	if (frame.empty())
	exit(0);
	
//frame.rows : nombre de lignes de l'image
//frame.cols : nombre de colonnes de l'image
//frame.at<Vec3b>(i,j)[0] : niveau de bleu du pixel (i,j) 
//frame.at<Vec3b>(i,j)[1] : niveau de vert du pixel (i,j) 
//frame.at<Vec3b>(i,j)[2] : niveau de rouge du pixel (i,j) 

	//TODO



	return frame_out;
}


//---------------------contour------------------------

Mat contour(Mat frame)
{ 
	
	Mat frame_out,frame_grayt;
	
	cvtColor(frame,frame_grayt,COLOR_RGB2GRAY);
	frame_out.create(frame.rows,frame.cols,CV_8UC1);
	
	if (frame.empty())
	exit(0);
	
//frame.rows : nombre de lignes de l'image
//frame.cols : nombre de colonnes de l'image
//frame.at<Vec3b>(i,j)[0] : niveau de bleu du pixel (i,j) 
//frame.at<Vec3b>(i,j)[1] : niveau de vert du pixel (i,j) 
//frame.at<Vec3b>(i,j)[2] : niveau de rouge du pixel (i,j) 

	//TODO
	
	return frame_out;
}


