#include <iostream>
#include "opencv2\core\core.hpp"
#include "opencv2\highgui.hpp"
#include "opencv2\imgproc.hpp"
#include "opencv2/opencv.hpp"


using namespace std;


int main()
{
	/// start video capture buffer
	cv::VideoCapture cap(CV_CAP_ANY);
	cap.set(CV_CAP_PROP_FRAME_WIDTH, 320);
	cap.set(CV_CAP_PROP_FRAME_HEIGHT, 240);
	if (!cap.isOpened())
		return -1;
	else
		cout << "Capture is opened" << endl;

	cv::Mat img;
	cv::namedWindow("video capture", CV_WINDOW_AUTOSIZE); // set the window
	cv::HOGDescriptor hog; /// create hog descriptor
	try {
		static vector<float> detectorArray = cv::HOGDescriptor::getDefaultPeopleDetector();
		hog.setSVMDetector(detectorArray);
	}
	catch (cv::Exception & e) {
		cout << e.msg << endl;
	}
	vector<cv::Rect> found, found_filtered;
	size_t i, j;
	cv::Rect r;

	while (true)
	{
		found.clear();
		found_filtered.clear();
		cap >> img;
		//img = cv::imread("E:\\Photos\\walkBarry8.jpg");
		//resize(img, img, cv::Size( 600, 480)); // resize the image for faster processing
		if (!img.data)
			continue;

		hog.detectMultiScale(img, found, 0, cv::Size(8, 8), cv::Size(32, 32), 1.05, 2); // start detection 

		for (i = 0; i< found.size(); i++)
		{
			r = found[i];
			for (j = 0; j<found.size(); j++)
				if (j != i && (r & found[j]) == r)
					break;
			if (j == found.size())
				found_filtered.push_back(r);
		}
		/// draw rect 

		for (i = 0; i<found_filtered.size(); i++)
		{
			/// shrink the rectangle as we get large 
			/// rectangles from detectMultiScale
			r = found_filtered[i];
			r.x += cvRound(r.width*0.1);
			r.width = cvRound(r.width*0.8);
			r.y += cvRound(r.height*0.06);
			r.height = cvRound(r.height*0.9);
			rectangle(img, r.tl(), r.br(), cv::Scalar(0, 255, 0), 2);
		}
		cv::imshow("video capture", img);
		//cv::imwrite("E:\\Photos\\goodDay everyone result.jpg", img);
		//cv::resizeWindow("video capture",800,600);
		if (cv::waitKey(20) >= 0)
			break;
		//cv::waitKey();
	}
	return 0;
}