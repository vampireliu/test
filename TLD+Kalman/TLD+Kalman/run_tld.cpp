#include <opencv2/opencv.hpp>
#include "tld_utils.h"
#include <iostream>
#include <sstream>   
//C++中的sstream类，提供了程序和string对象之间的I/O，可以通过
//ostringstream和istringstream两个类来声明对象，分别对应输出流和输入流

#include "TLD.h"
#include <stdio.h>
#include<sys/timeb.h>
using namespace cv;
using namespace std;
//Global variables
//全局变量
Rect box;
bool drawing_box = false;
bool gotBB = false;
bool tl = true;
bool rep = false;
bool fromfile=false;
string video;


//函数功能，从txt文件中读取box数据到全局变量box上
/**********************************************************
	读取记录bounding box的文件，获得bounding box的四个参数：左上角
	坐标（x，y）和宽高
***********************************************************/
void readBB(char* file){
  ifstream bb_file (file);   //创建文件输入流
  string line;
  getline(bb_file,line); //获取文件中的第一行
						 //将输入流bb_file中读到的字符存入string
						 //对象line中，终结符默认为'\n'
  istringstream linestream(line); //istringstream对象可以绑定一行
								  //字符串，然后以空格为分隔符把该行
								  //分隔开来
  string x1,y1,x2,y2;
  getline (linestream,x1, ','); //将输入流linestream读到的字符存入
  getline (linestream,y1, ','); //定义的x1,y1,x2,y2字符串变量中
  getline (linestream,x2, ','); //直到遇到终结符‘，’终止读取。
  getline (linestream,y2, ',');
  int x = atoi(x1.c_str());//把字符串转换成整型数
  int y = atoi(y1.c_str());// = (int)file["bb_y"];
  int w = atoi(x2.c_str())-x;// = (int)file["bb_w"];
  int h = atoi(y2.c_str())-y;// = (int)file["bb_h"];
  box = Rect(x,y,w,h);
}

/****************************************************
   bounding box mouse callback
   鼠标点击事件响应，用来返回人为设定的初始box
*****************************************************/
void mouseHandler(int event, int x, int y, int flags, void *param){
  switch( event ){
  case CV_EVENT_MOUSEMOVE:
    if (drawing_box){
        box.width = x-box.x;
        box.height = y-box.y;
    }
    break;
  case CV_EVENT_LBUTTONDOWN:
    drawing_box = true;
    box = Rect( x, y, 0, 0 );
    break;
  case CV_EVENT_LBUTTONUP:
    drawing_box = false;
    if( box.width < 0 ){
        box.x += box.width;
        box.width *= -1;
    }
    if( box.height < 0 ){
        box.y += box.height;
        box.height *= -1;
    }
    gotBB = true;  //已经获得bounding box
    break;
  }
}

void print_help(char** argv){
  printf("use:\n     %s -p /path/parameters.yml\n",argv[0]);
  printf("-s    source video\n-b        bounding box file\n-tl  track and learn\n-r     repeat\n");
}

int main(int argc, char * argv[])
{
  VideoCapture capture(0);
  FileStorage fs; //存储系统用到的参数，从parameters.yml获得
  ofstream processfream;
  processfream.open("processfream_time.txt");

//  read_options(argc,argv,capture,fs);  //读取参数
  fs.open("parameters.yml", FileStorage::READ);
 
 
  cvNamedWindow("TLD",CV_WINDOW_AUTOSIZE);
  if(!gotBB)
  {
	  cvSetMouseCallback( "TLD", mouseHandler, NULL );
  }


  //TLD framework
  TLD tld;
  //Read parameters file
  tld.read(fs.getFirstTopLevelNode());
  Mat frame;
  Mat last_gray;
  Mat first;
  if (fromfile){ //如果指定为从文件中读取
	  capture >> frame; //读取视频帧
      cvtColor(frame, last_gray, CV_RGB2GRAY); //转换为灰度图像
      frame.copyTo(first);  //拷贝作为第一帧
  }
  else{
	  capture.open("C://Users//Administrator//Desktop//正在学习。。。//datasets//car.mpg");  //C://Users//Administrator//Desktop//david.mpg"开启摄像头
	  if (!capture.isOpened())  //判断摄像头是否打开
	  {
			cout << "capture device failed to open!" << endl;
			return 1;
	  }

//      capture.set(CV_CAP_PROP_FRAME_WIDTH,340); //设置获取的图像大小为320*240
//      capture.set(CV_CAP_PROP_FRAME_HEIGHT,240);
  }
  
  ///Initialization
GETBOUNDINGBOX: //标号：获取bounding box
  while(!gotBB)
  {
    if (!fromfile){
      capture >> frame;
	  
    }
    else
      first.copyTo(frame);
    cvtColor(frame, last_gray, CV_RGB2GRAY);
    drawBox(frame,box);  //把boundbing box 画出来
    imshow("TLD", frame);

	int delay = 30;
	if (delay >= 0 && waitKey(delay) >= 0)
		waitKey(0);

    if (cvWaitKey(33) == 'q')
	    return 0;
  }
  //由于图相片（min_win为15*15像素）是在bounding box中采样得到的。所以box必须比
  //min_win要大
  if (min(box.width,box.height)<(int)fs.getFirstTopLevelNode()["min_win"])
  {
      cout << "Bounding box too small, try again." << endl;
      gotBB = false;
      goto GETBOUNDINGBOX;
  }
  //Remove callback
  cvSetMouseCallback( "TLD", NULL, NULL );//如果已经获得第一帧用户框定的box了就取消
										  //鼠标响应
  printf("Initial Bounding Box = x:%d y:%d h:%d w:%d\n",box.x,box.y,box.width,box.height);
  //Output file
  FILE  *bb_file = fopen("bounding_boxes.txt","w");

  
  //TLD initialization
  //TLD 进行初始化
  tld.init(last_gray,box,bb_file);

  ///Run-time
  Mat current_gray;
  BoundingBox pbox;
  vector<Point2f> pts1;
  vector<Point2f> pts2;
  bool status=true;  //记录跟踪成功与否的状态 lastbox been found
  int frames = 1;    //记录已过去的帧数
  int detections = 1; //记录成功检测到的目标box数目
REPEAT:
  while(capture.read(frame)){

    //get frame
    cvtColor(frame, current_gray, CV_RGB2GRAY);
    //Process Frame
	struct timeb start, end;
	ftime(&start);
    tld.processFrame(last_gray,current_gray,pts1,pts2,pbox,status,tl,bb_file);
	ftime(&end);
	std::cout << "processframe Process Time: " << (end.time - start.time) * 1000 + (end.millitm - start.millitm) << " ms" << endl;
	processfream << "frames : " << frames << "processframe Process Time: " << (end.time - start.time) * 1000 + (end.millitm - start.millitm) << "ms" << endl;


    //Draw Points
    if (status){ //如果跟踪成功
      drawPoints(frame,pts1);
      drawPoints(frame,pts2,Scalar(0,255,0)); //当前的特征点用绿色点表示
      drawBox(frame,pbox);
      detections++;
    }
    //Display
	char strFrame[20];
	sprintf(strFrame, "#%d ",frames) ;
	putText(frame,strFrame,cvPoint(0,20),2,1,CV_RGB(25,200,25));
    imshow("TLD", frame);
    //swap points and images
    swap(last_gray,current_gray); //STL函数swap()用来交换两对象的值。其泛型化版本
								  //定义于<algorithm>
    pts1.clear();
    pts2.clear();
    frames++;
    printf("Detection rate: %d/%d\n",detections,frames);
	if (status)
	{
		FileStorage boundingbox("newboundingbox.txt", FileStorage::APPEND);
		boundingbox << "framenumber" << frames << "x" << pbox.x << "y" << pbox.y << "width" << pbox.width << "height" << pbox.height;
	}
    if (cvWaitKey(33) == 'q')
      break;
  }
  if (rep){
    rep = false;
    tl = false;
    fclose(bb_file);
    bb_file = fopen("final_detector.txt","w");
    //capture.set(CV_CAP_PROP_POS_AVI_RATIO,0);
    capture.release();
    capture.open(video);
    goto REPEAT;
  }
  fclose(bb_file);
  return 0;
}
