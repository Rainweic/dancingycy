#include <cstdio>
#include <iostream>
#include <algorithm>
#include <cstring>
#include <cmath>
#include <queue>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/stitching/detail/blenders.hpp>

using namespace cv;
const int N=2000;
const int DEBUG=1;
const int C=100,delC=5;

class SC{
public:
	SC();
	~SC();

	void feed(Mat &_img,Mat &_mask);
	void doit(int dx,int dy,int _mode);
	Mat &getRes();

private:
	Mat img;
	Mat simg;
	Mat mask;
	int row,col;
	int srow,scol;
	int f[N][N];
	int g[N][N];
	int bo[N][N];
	int seam[N][N];
	int pos[N][N][2];
	int from[N][N];
	int val[N][N];
	int val2[N][N];
	Mat grad;
	int mode;
	int nMask;
	
	void init();
	void makeNMask();
	void countGrad();
	void clean();
	void drawImg(int dx,int dy);
	int large(int x1,int x2,int y1,int y2);
	void makeBo(int t,int s,int p);
	void delRow(int t);
	void addRow(int t);
	void delCol(int t);
	void addCol(int t);
	void printSeam();
};