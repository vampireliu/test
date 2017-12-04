#include "tld_utils.h"
using namespace cv;
using namespace std;

/*
	vector是C++标准模板库STL中的部分内容，它是一个多功能的，能够操作多种数据结构
	和算法的模板类和函数库。vector之所以被认为是一个容器，是因为它能够像容器一样存
	放各种类型的对象，简单的说，vector是一个能够存放任意类型的动态数组，能够增加和
	压缩数据。为了能够使用vector，必须在头文件中包含#include<vector>
*/
void drawBox(Mat& image, CvRect box, Scalar color, int thick){

  rectangle( image, cvPoint(box.x, box.y), 
	  cvPoint(box.x+box.width,box.y+box.height),color, thick);
} 

//函数cvRound cvFloor cvCeil用一种舍入方法将输入浮点数转换成整数。
//cvRound四舍五入取整，cvFloor下取整， cvCeil上取整
void drawPoints(Mat& image, vector<Point2f> points,Scalar color){
  for( vector<Point2f>::const_iterator iterator = points.begin(), ie = points.end(); iterator != ie; ++iterator )
      {
      //Point center( cvRound(iterator->x ), cvRound(iterator->y));
	  //circle(Mat& img, Point center, int radius, 
	         //const Scalar& color, int thickness=1, int lineType=8, int shift=0)
      circle(image,*iterator,2,color,1);
      }
}

Mat createMask(const Mat& image, CvRect box){
  Mat mask = Mat::zeros(image.rows,image.cols,CV_8U);
  drawBox(mask,box,Scalar::all(255),CV_FILLED);
  return mask;
}

/*
	STL中的nth_element()方法找出一个数列中排名第n的那个数对于数列a[0:len-1]将第n大的数字
	，排在a[n],同时a[0:n-1]都小于a[n],a[n+1:len-1]都大于a[n]，但a[n]左右的这两个序列不
	一定是有序的。用在中值流跟踪算法中，寻找中值。
*/
float median(vector<float> v)
{
    int n = floor(double(v.size() / 2));
    nth_element(v.begin(), v.begin()+n, v.end());
    return v[n];
}

/*
	1、<algorithm> //random_shuffle的头文件
	shuffle洗牌:首先简单介绍一个扑克洗牌的方法，假设一个数组poker[52]中存有一副扑克
	牌1-52的牌点值，使用一个for循环遍历这个数组，每次循环都生成一个[0,52)之间的随机数
	RandNum,以RandNum为数组下标，把当前下标对应的值和RandNUm对应位置的值交换，循环
	结束。每个牌都与某个位置交换了一次，这样一副牌就被打乱了。理解代码如下：

	for(int i=0; i<52; ++i)
	{
		int RandNum=rand()%52;
		int tmp=poker[i];
		poker[i]=poker[RandNum];
		poker[RandNum]=tmp;
	}

	2、需要指定范围内的随机数，传统的方法是使用ANSI C的函数random()，然后格式化结果
	以便结果是落在指定的范围内。但是，使用这个方法至少有两个缺点。做格式化时，结果常常
	是扭曲的，且只支持整型数。
	3、C++中提供了更好的解决方法，那就是STL中的random_shuffle()算法。产生指定范围
	内的随机元素的最佳方法。它是创建一个顺序序列，在这个顺序序列中含有指定范围的所有值，
	例如，如果需要产生100个0到99之间的数，那么就创建一个向量并用100个按升序排列的填充
	向量，填充完向量后，用random——shuffle()算法打乱元素的排列顺序。
	默认的random_shuffle中，被操作序列的index与rand()%N两个位置的值交换，来达到乱
	序的目的.index_shuffle()用于产生指定范围[begin:end]的随机数，返回随机数数组。
*/
vector<int> index_shuffle(int begin,int end){
  vector<int> indexes(end-begin);
  for (int i=begin;i<end;i++){
    indexes[i]=i;
  }
  random_shuffle(indexes.begin(),indexes.end());
  return indexes;
}

