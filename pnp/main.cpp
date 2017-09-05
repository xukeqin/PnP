#include <Eigen/Core>
#include <Eigen/LU>
#include<opencv2/opencv.hpp>  
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/core/eigen.hpp>
#include <Eigen/Dense>
#include "Marker.h"
using namespace std;
using namespace cv;

vector<cv::Point3f> m_markerCorners3d;
vector<cv::Point2f> m_markerCorners2d;
Size markerSize(100,100);

Mat camMatrix;
Mat distCoeff;
float m_minContourLengthAllowed=30;

//计算周长
float perimeter(const std::vector<cv::Point2f> &a)
{
	float sum = 0, dx, dy;

	for (size_t i = 0; i<a.size(); ++i)
	{
		size_t i2 = (i + 1) % a.size();

		dx = a[i].x - a[i2].x;
		dy = a[i].y - a[i2].y;

		sum += sqrt(dx*dx + dy*dy);
	}

	return sum;
}


void findMarkerCandidates(const std::vector<std::vector<cv::Point>>& contours, std::vector<Marker>& detectedMarkers)
{
	std::vector<cv::Point>  approxCurve;
	std::vector<Marker> possibleMarkers;
	// For each contour, analyze if it is a paralelepiped likely to be the marker
	for (size_t i = 0; i<contours.size(); ++i)
	{
		// 拟合多边形轮廓
		cv::approxPolyDP(contours[i], approxCurve, double(contours[i].size())*0.05, true);                  //input arrary点集    approxCurve 拟合的点   第三个参数为精度  第四个表示为闭合

		// 找寻四边形轮廓
		if (approxCurve.size() != 4)
			continue;

		// 判断是不是凸多边形
		if (!cv::isContourConvex(approxCurve))
			continue;

		//四个点之间距离的最小值滤除标记
		float minDist = 1e10;
		for (int i = 0; i<4; ++i)
		{
			cv::Point vec = approxCurve[i] - approxCurve[(i + 1) % 4];
			float squaredDistance = vec.dot(vec);
			minDist = std::min(minDist, squaredDistance);//取间距的最小值
		}

		// 预先设定的最小标记值
		if (minDist < m_minContourLengthAllowed)
			continue;

		//滤除条件满足的所有轮廓进行检验     
		Marker m;
		for (int i = 0; i<4; ++i)
		{
			m.points.push_back(cv::Point2f(approxCurve[i].x, approxCurve[i].y));
		}
		//逆时针对点进行排序：连接第一个点和第二个点，如果第三个点在第三个点在右边则是逆时针的
		cv::Point v1 = m.points[1] - m.points[0];
		cv::Point v2 = m.points[2] - m.points[0];
		double o = (v1.x * v2.y) - (v1.y * v2.x);
		//如果在左边应该交换第二个和第四个点便达到效果
		if (o  < 0.0)
		{
			std::swap(m.points[1], m.points[3]);
		}
		possibleMarkers.push_back(m);
	}
	//去除角点过为接近的轮廓
	//定义为pair的容器，两个选择性的滤除其中一个
	std::vector< std::pair<int, int> > tooNearCandidates;
	for (size_t i = 0; i<possibleMarkers.size(); ++i)
	{
		const Marker& m1 = possibleMarkers[i];
		//计算边长的均值
		for (size_t j = i + 1; j<possibleMarkers.size(); ++j)
		{
			const Marker& m2 = possibleMarkers[j];
			float distSquared = 0;
			for (int c = 0; c<4; ++c)
			{
				cv::Point v = m1.points[c] - m2.points[c];
				distSquared += v.dot(v);
			}
			distSquared /= 4;
			//改变的最近边长
			if (distSquared < 50)
			{
				tooNearCandidates.push_back(std::pair<int, int>(i, j));
			}
		}
	}

	//标记类型的的vertor作为标志位
	//去除重影，要大的 - -
	std::vector<bool> removalMask(possibleMarkers.size(), false);
	//
	for (size_t i = 0; i<tooNearCandidates.size(); ++i)
	{
		float p1 = perimeter(possibleMarkers[tooNearCandidates[i].first].points);
		float p2 = perimeter(possibleMarkers[tooNearCandidates[i].second].points);
		size_t removalIndex;
		if (p1 > p2)
			removalIndex = tooNearCandidates[i].second;
		else
			removalIndex = tooNearCandidates[i].first;

		//选择其中的一个不滤除
		removalMask[removalIndex] = true;
	}

	//返回经过所有滤除步骤后的满足条件的Marker
	detectedMarkers.clear();
	for (size_t i = 0; i<possibleMarkers.size(); ++i)
	{
		if (!removalMask[i])
			detectedMarkers.push_back(possibleMarkers[i]);
	}
}


void detectMarkers(const cv::Mat& grayscale, std::vector<Marker>& detectedMarkers)
{
	cv::Mat canonicalMarker;
	std::vector<Marker> goodMarkers;

	//////////////透视变换找寻goodmarker////////////////////
	for (size_t i = 0; i<detectedMarkers.size(); ++i)
	{
		Marker& marker = detectedMarkers[i];
		//得到透视变换的变换矩阵，通过四个顶点得到。第一个参数是标记在空间中的坐标，第二个参数是四个顶点的坐标。
		cv::Mat M = cv::getPerspectiveTransform(marker.points, m_markerCorners2d);   //变换关系矩阵

		// Transform image to get a canonical marker image
		// 透视变换成方形图像
		cv::warpPerspective(grayscale, canonicalMarker, M, markerSize);           //真正的变换 ，canonicalmarker是图
		threshold(canonicalMarker, canonicalMarker, 125, 255, THRESH_BINARY | THRESH_OTSU); //OTSU determins threshold automatically.

		//显示变换成功的图像
		//imshow("Gray Image1",canonicalMarker);
		int nRotations;

		//cout<<"canonicalMarker"<<canonicalMarker.size()<<endl;
		// 标记编码识别重要函数：亚像素级别的检测二维码包含的信息————>//
		int id = Marker::getMarkerId(canonicalMarker, nRotations);
		//判断是否符合预定二维码信息
		if (id != -1)
		{
			marker.id = id;

			//sort the points so that they are always in the same order no matter the camera orientation
			std::rotate(marker.points.begin(), marker.points.begin() + 4 - nRotations, marker.points.end());
			goodMarkers.push_back(marker);
			//cout << goodMarkers.data<< endl;
		}
	}

	//细化角点
	if (goodMarkers.size() > 0)
	{
		std::vector<cv::Point2f> preciseCorners(4 * goodMarkers.size());
		for (size_t i = 0; i<goodMarkers.size(); ++i)
		{
			Marker& marker = goodMarkers[i];
			for (int c = 0; c<4; ++c)
			{
				preciseCorners[i * 4 + c] = marker.points[c];
				//整数输出
				//cout << preciseCorners[i * 4 + c] << endl;
			}
		}

		//细化角点函数
		cv::cornerSubPix(grayscale, preciseCorners, cvSize(5, 5), cvSize(-1, -1), cvTermCriteria(CV_TERMCRIT_ITER, 30, 0.1));

		//copy back
		//将细化位置复制给标记角点
		for (size_t i = 0; i<goodMarkers.size(); ++i)
		{
			Marker& marker = goodMarkers[i];

			for (int c = 0; c<4; ++c)
			{
				marker.points[c] = preciseCorners[i * 4 + c];
				//浮点数输出
				//cout << marker.points[c] << endl;
			}
		}
	}

	//赋值由于下一步参数计算
	detectedMarkers = goodMarkers;
	//cout<<"detectedMarkers.size()"<<detectedMarkers.size()<<endl;
}


void estimatePosition(std::vector<Marker>& detectedMarkers)
{

	//使用for循环是在
	for (size_t i = 0; i<detectedMarkers.size(); ++i)
	{
		Marker& m = detectedMarkers[i];
		cv::Mat Rvec;
		cv::Mat_<float> Tvec;
		cv::Mat raux, taux;
		//
		cv::solvePnP(m_markerCorners3d, m.points, camMatrix, distCoeff, raux, taux, false, CV_P3P);

		raux.convertTo(Rvec, CV_32F);    //旋转向量
		taux.convertTo(Tvec, CV_32F);   //平移向量

		cv::Mat_<float> rotMat(3, 3);
		cv::Rodrigues(Rvec, rotMat);
		// Copy to transformation matrix

		("S dcode");
		// Since solvePnP finds camera location, w.r.t to marker pose, to get marker pose w.r.t to the camera we invert it.
		//std::cout << " Tvec ( X<-, Y ^, Z * ） ：" << Tvec.rows << "x" << Tvec.cols << std::endl;
		//std::cout << Tvec <<endl;		//平移矩阵
		//std::cout << " Rvec ( X<-, Y ^, Z * ） ：" << Rvec.rows << "x" << Rvec.cols << std::endl;
		//std::cout << rotMat << endl;      //旋转矩阵
		//std::cout << camMatrix << endl;      //旋转矩阵
		//std::cout << distCoeff << endl;      //旋转矩阵

		float theta_z = atan2(rotMat[1][0], rotMat[0][0])*57.2958;
		float theta_y = atan2(-rotMat[2][0], sqrt(rotMat[2][0] * rotMat[2][0] + rotMat[2][2] * rotMat[2][2]))*57.2958;
		float theta_x = atan2(rotMat[2][1], rotMat[2][2])*57.2958;

		//void cv::cv2eigen(const Mat &rotMat, Eigen::Matrix< float, 1, Eigen::Dynamic > &R_n);

		Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> R_n;
		Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> T_n;
		cv2eigen(rotMat, R_n);
		cv2eigen(Tvec, T_n);
		Eigen::Vector3f P_oc;

		P_oc = -R_n.inverse()*T_n;


		std::cout << "世界坐标" << P_oc << std::endl;
		//std::cout << "旋转向量" << raux << std::endl;
		//std::cout << "像素坐标" << m.points << std::endl;
		//std::cout << "输入的世界坐标" << m_markerCorners3d << std::endl;

		//std::cout << "\nX ：" << theta_x << std::endl;
		//std::cout << "Y ：" << theta_y << std::endl;
		//std::cout << "Z ：" << theta_z << std::endl;

	}
}



//寻找轮廓主函数
void Marker_Detection(Mat& img, vector<Marker>& detectedMarkers)
{
	Mat imgGray;
	Mat imgByAdptThr;
	vector<vector<Point>> contours;


	//将图像转为灰度图
	cvtColor(img, imgGray, CV_BGRA2GRAY);

	//二值化
	threshold(imgGray, imgByAdptThr, 160, 255, THRESH_BINARY_INV);

	//开运算和闭运算
	morphologyEx(imgByAdptThr, imgByAdptThr, MORPH_OPEN, Mat());
	morphologyEx(imgByAdptThr, imgByAdptThr, MORPH_CLOSE, Mat());

	//轮廓检测
	std::vector<std::vector<cv::Point> > allContours;
	cv::findContours(imgByAdptThr, allContours, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);
	contours.clear();
	for (size_t i = 0; i<allContours.size(); ++i)
	{
		int contourSize = allContours[i].size();
		if (contourSize > 4)
		{
			contours.push_back(allContours[i]);
		}
	}
	//判断非0
	if (contours.size())
	{
		//寻找符合条件的轮廓
		findMarkerCandidates(contours, detectedMarkers);
	}

	//判断非0
	if (detectedMarkers.size())
	{
		//检测标记二维码信息
		detectMarkers(imgGray, detectedMarkers);//灰度图中找寻信息

		//计算坐标
		estimatePosition(detectedMarkers);
	}
}




int main()
{
	m_markerCorners3d.push_back(cv::Point3f(-70.0f, -70.0f, 0));
	m_markerCorners3d.push_back(cv::Point3f(+70.0f, -70.0f, 0));    //左上角为原点
	m_markerCorners3d.push_back(cv::Point3f(+70.0f, +70.0f, 0));
	m_markerCorners3d.push_back(cv::Point3f(-70.0f, +70.0f, 0));

	m_markerCorners2d.push_back(cv::Point2f(0, 0));
	m_markerCorners2d.push_back(cv::Point2f(markerSize.width - 1, 0));
	m_markerCorners2d.push_back(cv::Point2f(markerSize.width - 1, markerSize.height - 1));
	m_markerCorners2d.push_back(cv::Point2f(0, markerSize.height - 1));

	camMatrix = (Mat_<double>(3, 3) << 598.29493, 0, 304.76898, 0, 597.56086, 233.34673, 0, 0, 1);
	distCoeff = (Mat_<double>(5, 1) << -0.53572,1.35993,-0.00244,0.00620,0.00000);

	int color_width = 1920; //color
	int color_height = 1080;

	VideoCapture capture(1);
	Mat frame;
	uchar *ppbuffer = frame.ptr<uchar>(0);

	while(1)
	{
		capture >> frame;

		Mat colorImg(color_height, color_width, CV_8UC4, (void*)ppbuffer);
		//复制给全局变量
		//colorImg.copyTo(colorsrc);      //??????????????????    
		//单通道返回彩色图
		//Mat colorImg = cv::Mat::zeros(color_height, color_width, CV_8UC1);//the color image 
		//colorImg = ConvertMat(pBuffer_color, color_width, color_height);

		//窗口显示

		
			//使用二维码查找
		vector<Marker> detectedM;
		Marker_Detection(frame, detectedM);
		for (int marknum = 0; marknum < detectedM.size(); ++marknum)
		{
			vector<Point3f> Coord;
			int validnum = 0;
			for (int c = 0; c < 4; ++c)
			{
				//输出角点坐标
				//cout << "(x, y)    =\t" << detectedMarkers[marknum].points[c] << endl;
				Point tempPoint = detectedM[marknum].points[c];

				//标记角点
				circle(frame, detectedM[marknum].points[c], 5, Scalar(0, 0, 255), -1, 2);
				line(frame, detectedM[marknum].points[(c + 1) % 4], detectedM[marknum].points[c], Scalar(0, 255, 0), 1, 8);
			}
		}

		
		//尾处理

		cv::imshow("suoxiao", frame);

		waitKey(1);

	}

}
