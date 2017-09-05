#ifndef Example_MarkerBasedAR_Marker_hpp
#define Example_MarkerBasedAR_Marker_hpp
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
using namespace std;
using namespace cv;

class Marker
{  
public:
	Marker();
	~Marker();
	friend bool operator<(const Marker &M1,const Marker&M2);
	friend std::ostream & operator<<(std::ostream &str,const Marker &M);
	//opencvÐý×ªº¯Êý
	static cv::Mat rotate(cv::Mat  in);
	static int hammDistMarker(cv::Mat bits);
	static int mat2id(const cv::Mat &bits);
	static int getMarkerId(cv::Mat &in,int &nRotations);
	//
	int id;
	std::vector<cv::Point2f> points;
};
#endif