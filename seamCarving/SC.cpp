#include "SC.h"

SC::SC(){
	memset(val,0,sizeof(val));
	memset(val2,0,sizeof(val2));
}

SC::~SC(){

}

void SC::feed(Mat &_img,Mat &_mask){
	img=_img.clone();
	mask=_mask.clone();
	row=img.rows;
	col=img.cols;
	memset(val2,0,sizeof(val2));
}

void SC::doit(int dx,int dy,int _mode){
	mode=_mode;
	memset(seam,0,sizeof(seam));
	init();
	if (mode==1){
		makeNMask();
		int snMask=0;
		while (nMask!=snMask){
			snMask=nMask;
			delRow(0);
			delCol(0);
		}
		printSeam();
		return;
	}
	while (dx>0 || dy>0){
		if (dx>0){
			addRow(dx);
			--dx;
		}
		if (dy>0){
			addCol(dy);
			--dy;
		}
	}
	init();
	while (dx<0 || dy<0){
		if (dx<0){
			delRow(dx);
			++dx;
		}
		if (dy<0){
			delCol(dy);
			++dy;
		}
	}
	printSeam();
	/*
	if (dx>0){
		for (int i=0;i<dx;i++) addRow();
	}else{
		for (int i=0;i<-dx;i++) delRow();
	}
	memset(val,0,sizeof(val));
	memset(val2,0,sizeof(val2));
	if (dy>0){
		for (int i=0;i<dy;i++) addCol();
	}else{
		for (int i=0;i<-dy;i++) delCol();
	}*/
}

Mat& SC::getRes(){
	return img;
}

void SC::init(){
	simg=img.clone();
	srow=row;
	scol=col;
	for (int i=0;i<row;i++)
		for (int j=0;j<col;j++){
			pos[i][j][0]=i;
			pos[i][j][1]=j;
		}
}

void SC::makeNMask(){
	nMask=0;
	for (int i=0;i<row;i++)
		for (int j=0;j<col;j++)
		if (mask.at<Vec3b>(i,j)[0]) ++nMask;
}

void SC::countGrad(){
  Mat src, src_gray;
  char* window_name = "Sobel Demo - Simple Edge Detector";
  int scale = 1;
  int delta = 0;
  int ddepth = CV_16S;

  /// 装载图像
  src = img.clone();
  grad = img.clone();

  GaussianBlur( src, src, Size(3,3), 0, 0, BORDER_DEFAULT );

  /// 转换为灰度图
  cvtColor( src, src_gray, CV_RGB2GRAY );

  /// 创建显示窗口
  //if (DEBUG) namedWindow( window_name, CV_WINDOW_AUTOSIZE );

  /// 创建 grad_x 和 grad_y 矩阵
  Mat grad_x, grad_y;
  Mat abs_grad_x, abs_grad_y;

  /// 求 X方向梯度
  //Scharr( src_gray, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );
  Sobel( src_gray, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
  convertScaleAbs( grad_x, abs_grad_x );

  /// 求Y方向梯度
  //Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
  Sobel( src_gray, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
  convertScaleAbs( grad_y, abs_grad_y );

  /// 合并梯度(近似)
  addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);
  if (DEBUG){
	  //imshow(window_name, grad);
	  //waitKey(1);
  }
  
  for (int i=0;i<row;i++)
	  for (int j=0;j<col;j++){
		  //printf("%d %d\n",i,j);
		  if (i==679 && j==681){
			  i=679;
		  }
		  val[i][j]=val2[i][j]+grad.at<uchar>(i,j)*((!i || !j || i==row-1 || j==col-1) ? 2 : 1);
		  if (i) val[i][j]+=val2[i-1][j];
		  if (j) val[i][j]+=val2[i][j-1];
		  if (i<row-1) val[i][j]+=val2[i+1][j];
		  if (j<col-1) val[i][j]+=val2[i][j+1];
		  if (mode==1 && mask.at<Vec3b>(i,j)[0])
			  val[i][j]-=50;
	  }
  //waitKey(1);
}

void SC::clean(){
	memset(f,0,sizeof(f));
	memset(g,0,sizeof(g));
	memset(bo,0,sizeof(bo));
}

void SC::drawImg(int dx,int dy){
	int srow=row+dx,scol=col+dy;
	Mat res(srow,scol,CV_8UC3);
	Mat res_mask(srow,scol,CV_8UC3,Scalar(0,0,0));
	Mat res2=img.clone();
	
	for (int i=0;i<row;i++)
		for (int j=0;j<col;j++)
		if (bo[i][j]){
			res2.at<Vec3b>(i,j)[0]=0;
			res2.at<Vec3b>(i,j)[1]=0;
			res2.at<Vec3b>(i,j)[2]=255;
		}
	//imshow("res2",res2);

	if (dx==-1 && dy==0){
		for (int j=0;j<scol;j++)
			for (int i=0,b=0;i<srow;i++){
				if (bo[i][j]){
					if (mode && mask.at<Vec3b>(i,j)[0]) --nMask;
					seam[pos[i][j][0]][pos[i][j][1]]=1;
					b=1;
				}
				for (int k=0;k<3;k++){
					res.at<Vec3b>(i,j)[k]=img.at<Vec3b>(i+b,j)[k];
					pos[i][j][0]=pos[i+b][j][0];
					pos[i][j][1]=pos[i+b][j][1];
					if (mode) res_mask.at<Vec3b>(i,j)[k]=mask.at<Vec3b>(i+b,j)[k];
				}
			}
	}

	if (dx==0 && dy==-1){
		for (int i=0;i<srow;i++)
			for (int j=0,b=0;j<scol;j++){
				if (bo[i][j]){
					if (mode && mask.at<Vec3b>(i,j)[0]) --nMask;
					seam[pos[i][j][0]][pos[i][j][1]]=1;
					b=1;
				}
				for (int k=0;k<3;k++){
					res.at<Vec3b>(i,j)[k]=img.at<Vec3b>(i,j+b)[k];
					pos[i][j][0]=pos[i][j+b][0];
					pos[i][j][1]=pos[i][j+b][1];
					if (mode) res_mask.at<Vec3b>(i,j)[k]=mask.at<Vec3b>(i,j+b)[k];
				}
			}
	}

	if (dx==1 && dy==0){
		for (int j=0;j<scol;j++){
			for (int i=0,b=0,ss=0;i<srow;i++){
				int sss=seam[i][j];
				if (bo[i][j] && i){
					b=-1;
					seam[i][j]=2;
					for (int k=0;k<3;k++){
						res.at<Vec3b>(i,j)[k]=(img.at<Vec3b>(i,j)[k]+img.at<Vec3b>(i+b,j)[k])/2;
						res_mask.at<Vec3b>(i,j)[k]=mask.at<Vec3b>(i,j)[k];
					}
				}else for (int k=0;k<3;k++){
					res.at<Vec3b>(i,j)[k]=img.at<Vec3b>(i+b,j)[k];
					res_mask.at<Vec3b>(i,j)[k]=mask.at<Vec3b>(i+b,j)[k];
					if (b) seam[i][j]=ss;
				}
				if (bo[i][j]) b=-1;
				ss=sss;
			}
			for (int i=srow-1;i>=0;i--){
				val2[i][j]=val2[i-1][j];
				if (val2[i][j]){
					val2[i][j]-=delC;
					if (val2[i][j]<0) val2[i][j]=0;
				}
				if (bo[i][j]){
					val2[i][j]=C;
					break;
				}
			}
		}
	}

	if (dx==0 && dy==1){
		for (int i=0;i<srow;i++){
			//printf("%d\n",i);
			for (int j=0,b=0,ss=0;j<scol;j++){
				int sss=seam[i][j];
				if (bo[i][j] && j){
					b=-1;
					seam[i][j]=2;
					for (int k=0;k<3;k++){
						res.at<Vec3b>(i,j)[k]=(img.at<Vec3b>(i,j)[k]+img.at<Vec3b>(i,j+b)[k])/2;
						res_mask.at<Vec3b>(i,j)[k]=mask.at<Vec3b>(i,j)[k];
					}
				}else{
					for (int k=0;k<3;k++){
						res.at<Vec3b>(i,j)[k]=img.at<Vec3b>(i,j+b)[k];
						res_mask.at<Vec3b>(i,j)[k]=mask.at<Vec3b>(i,j+b)[k];
					}
					if (b) seam[i][j]=ss;
				}
				if (bo[i][j]) b=-1;
				ss=sss;
			}
			for (int j=scol-1;j>=0;j--){
				val2[i][j]=val2[i][j-1];
				if (val2[i][j]){
					val2[i][j]-=delC;
					if (val2[i][j]<0) val2[i][j]=0;
				}
				if (bo[i][j]){
					val2[i][j]=C;
					break;
				}
			}
		}
	}

	imshow("res",res);
	waitKey(1);
	img=res.clone();
	mask=res_mask.clone();
}

int SC::large(int x1,int x2,int y1,int y2){
	return x1>x2 || x1==x2 && y1>y2;
}

void SC::makeBo(int t,int s,int p){
	if (t==0){  //横向缩放
		int st=0,ed=row/2;
		if (p%2) st=row/2,ed=row;
		for (int j=0;j<col;j++)
			for (int i=0;i<row;i++)
			if (j){
				if (mode!=2 || (!mask.at<Vec3b>(i,j)[0] && st<=i && i<=ed)){
					if (mode!=1 || mask.at<Vec3b>(i,j)[0]) g[i][j]=1;
					int sg;
					f[i][j]=val[i][j]+f[i][j-1];  //从i方向转移
					from[i][j]=i;
					sg=!g[i][j] && g[i][j-1];
					if (i)  //从i-1方向转移
					if (!s && large(!g[i][j] && g[i-1][j-1],sg,f[i][j],val[i][j]+f[i-1][j-1]) || s && f[i][j]<val[i][j]+f[i-1][j-1]){
						f[i][j]=val[i][j]+f[i-1][j-1];
						sg=!g[i][j] && g[i-1][j-1];
						from[i][j]=i-1;
					}
					if (i<row-1)  //从i+1方向转移
					if (!s && large(!g[i][j] && g[i+1][j-1],sg,f[i][j],val[i][j]+f[i+1][j-1]) || s && f[i][j]<val[i][j]+f[i+1][j-1]){
						f[i][j]=val[i][j]+f[i+1][j-1];
						sg=!g[i][j] && g[i+1][j-1];
						from[i][j]=i+1;
					}
					g[i][j]|=sg;
				}else g[i][j]=1,f[i][j]=1e9;
			}else{  //第一列
				f[i][j]=val[i][j];
				if (mode!=1 || mask.at<Vec3b>(i,j)[0]) g[i][j]=1;
			}

		int min=-1,sg=0,max=-1,si;
		for (int i=st;i<ed;i++){
			if (!s && (min==-1 || large(g[i][col-1],sg,min,f[i][col-1]))){  //选择最小能量点
				min=f[i][col-1];
				sg=g[i][col-1];
				si=i;
			}
			if (s && (max==-1 || max<f[i][col-1])){
				max=f[i][col-1];
				si=i;
			}
		}
		//si=rand()%row;
		for (int i=si,j=col-1;j>=0;j--){  //从最小能量点寻找seam
			bo[i][j]=1;
			i=from[i][j];
		}
	}else{  //纵向缩放
		int st=0,ed=col/2;
		if (p%2) st=col/2,ed=col;
		for (int i=0;i<row;i++)
			for (int j=0;j<col;j++)
			if (i){
				if (mode!=2 || (!mask.at<Vec3b>(i,j)[0] && st<=j && j<=ed)){
					if (mode!=1 || mask.at<Vec3b>(i,j)[0]) g[i][j]=1;
					int sg;
					f[i][j]=val[i][j]+f[i-1][j];  //从j方向转移
					from[i][j]=j;
					sg=!g[i][j] && g[i-1][j];
					if (j)  //从j-1方向转移
					if (!s && large(!g[i][j] && g[i-1][j-1],sg,f[i][j],val[i][j]+f[i-1][j-1]) || s && f[i][j]<val[i][j]+f[i-1][j-1]){
						f[i][j]=val[i][j]+f[i-1][j-1];
						sg=!g[i][j] && g[i-1][j-1];
						from[i][j]=j-1;
					}
					if (j<col-1)  //从j+1方向转移
					if (!s && large(!g[i][j] && g[i-1][j+1],sg,f[i][j],val[i][j]+f[i-1][j+1]) || s && f[i][j]<val[i][j]+f[i-1][j+1]){
						f[i][j]=val[i][j]+f[i-1][j+1];
						sg=!g[i][j] && g[i-1][j+1];
						from[i][j]=j+1;
					}
					g[i][j]|=sg;
				}else g[i][j]=1,f[i][j]=1e9;
			}else{  //第一行
				f[i][j]=val[i][j];
				if (mode!=1 || mask.at<Vec3b>(i,j)[0]) g[i][j]=1;
			}
	
		int min=-1,sg=0,max=-1,sj;
		for (int j=st;j<ed;j++){
			if (!s && (min==-1 || large(g[row-1][j],sg,min,f[row-1][j]))){
				min=f[row-1][j];
				sg=g[row-1][j];
				sj=j;
			}
			if (s && (max==-1 || max<f[row-1][j])){
				max=f[row-1][j];
				sj=j;
			}
		}
		//sj=rand()%col;
		for (int i=row-1,j=sj;i>=0;i--){
			bo[i][j]=1;
			j=from[i][j];
		}
	}
}

void SC::delRow(int t=0){
	countGrad();
	clean();
	makeBo(0,0,t);
	drawImg(-1,0);
	--row;
}

void SC::addRow(int t=0){
	countGrad();
	clean();
	makeBo(0,0,t);
	drawImg(1,0);
	++row;
}

void SC::delCol(int t=0){
	countGrad();
	clean();
	makeBo(1,0,t);
	drawImg(0,-1);
	--col;
}

void SC::addCol(int t=0){
	countGrad();
	clean();
	makeBo(1,0,t);
	drawImg(0,1);
	++col;
}

void SC::printSeam(){
	Mat res=simg.clone();
	for (int i=0;i<srow;i++)
		for (int j=0;j<scol;j++)
		if (seam[i][j]==1){
			res.at<Vec3b>(i,j)[0]=0;
			res.at<Vec3b>(i,j)[1]=0;
			res.at<Vec3b>(i,j)[2]=255;
		}else if (seam[i][j]==2){
			res.at<Vec3b>(i,j)[0]=255;
			res.at<Vec3b>(i,j)[1]=0;
			res.at<Vec3b>(i,j)[2]=0;
		}
	imwrite("seam.jpg",res);
}