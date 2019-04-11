#include "SC.h"

SC sc;

int main(){
	Mat mask=imread("mask.jpg");
	Mat img=imread("ycy.jpg");
	imshow("mask",mask);
	resize(mask,mask,Size(),0.5,0.5);
	resize(img,img,Size(),0.5,0.5);
	sc.feed(img,mask);
	sc.doit(200,200,2);
	Mat res=sc.getRes().clone();
	imwrite("res.jpg",res);
	/*
    VideoCapture cap;
    cap.open("ycy.mp4");
	
	Mat mask=imread("mask.jpg");
	string outputVideoPath = "..\\test.avi";
	//获取当前摄像头的视频信息
	cv::Size sWH = cv::Size(500,500);
	VideoWriter outputVideo;
	outputVideo.open(outputVideoPath, CV_FOURCC('M', 'P', '4', '2'), 25.0, sWH);

	int num=0;
    while(1)
    {
		printf("%d\n",num++);
		Mat img;
        cap>>img;
		resize(img,img,Size(281,500));
        if(img.empty()) break;

		sc.feed(img,mask);
		sc.doit(0,219,0);
		Mat res=sc.getRes().clone();
		outputVideo << res;
	}
	outputVideo.release();*/
}