/*
 * FerNNClassifier.h
 *
 *  Created on: Jun 14, 2011
 *      Author: alantrrs
 */

#include <opencv2/opencv.hpp>
#include <stdio.h>
class FerNNClassifier{
private: //下面这些参数通过程序开始运行时读入parameters.yml文件进行初始化
  float thr_fern;
  int structSize;
  int nstructs;
  float valid;
  float ncc_thesame;
  float thr_nn;//0.65
  int acum;
public:
  //Parameters
  float thr_nn_valid;

  void read(const cv::FileNode& file);
  void prepare(const std::vector<cv::Size>& scales);
  void getFeatures(const cv::Mat& image,const int& scale_idx,std::vector<int>& fern);
  void update(const std::vector<int>& fern, int C, int N);
  float measure_forest(std::vector<int> fern);
  void trainF(const std::vector<std::pair<std::vector<int>,int> >& ferns,int resample);
  void trainNN(const std::vector<cv::Mat>& nn_examples);
  void NNConf(const cv::Mat& example,std::vector<int>& isin,float& rsconf,float& csconf);
  void evaluateTh(const std::vector<std::pair<std::vector<int>,int> >& nXT,const std::vector<cv::Mat>& nExT);
  void show();
  //Ferns Members
  int getNumStructs(){return nstructs;}
  float getFernTh(){return thr_fern;}
  float getNNTh(){return thr_nn;}
  struct Feature //特征结构体
      {
          uchar x1, y1, x2, y2;
		  //无参数的构造函数，将x1,y1,x2,y2初始化为0
          Feature() : x1(0), y1(0), x2(0), y2(0) {}
		  //构造函数的重载，有参数情况下，将参数强制转换为uchar类型
          Feature(int _x1, int _y1, int _x2, int _y2)
          : x1((uchar)_x1), y1((uchar)_y1), x2((uchar)_x2), y2((uchar)_y2)
          {}

		  /*
		  1.下面的函数operator后面有两个括号表示函数的重载
		  2.函数的后面const表示该函数不能改变对象的成员变量
		  */
          bool operator ()(const cv::Mat& patch) const
          { 
			  /*
			  二维单通道元素可以用Mat::at(i, j)访问，i是行序号，j是列序号
			  返回的patch图像片在(y1,x1)和(y2, x2)点的像素比较值，返回0或者1 
			  */
			  return patch.at<uchar>(y1,x1) > patch.at<uchar>(y2, x2); 
		  }
      };
  std::vector<std::vector<Feature> > features; //Ferns features (one std::vector for each scale)
  std::vector< std::vector<int> > nCounter; //negative counter
  std::vector< std::vector<int> > pCounter; //positive counter
  std::vector< std::vector<float> > posteriors; //Ferns posteriors
  float thrN; //Negative threshold
  float thrP;  //Positive thershold
  //NN Members
  std::vector<cv::Mat> pEx; //NN positive examples
  std::vector<cv::Mat> nEx; //NN negative examples
};
