#include "tld_utils.h"
#include <opencv2/opencv.hpp>

//使用金字塔光流法跟踪，所以类成员变量中有很多是OpenCV中calcOpticalFlowPyLK()
//函数的参数
class LKTracker{
private:
  std::vector<cv::Point2f> pointsFB;
  cv::Size window_size;  //每个金字塔层的搜索窗口尺寸
  int level; //最大的金字塔层数
  std::vector<uchar> status; //容器数组，如果对应特征的光流被发现，数组
							 //中的每个元素都被设置为1，否则设置为0
  std::vector<uchar> FB_status;
  std::vector<float> similarity; //相似度
  std::vector<float> FB_error;   //Forward-Backward error方法，求FB_error
								 //的结果与原始位置的欧式距离作比较，
								 //把距离过大的跟踪结果舍弃
  float simmed;
  float fbmed; 
  /*
	TermCriteria模板类，取代了之前的CvTermCriteria，这个类是作为迭代算
	法的终止条件的，该类变量需要3个参数，一个是类型，第二个参数为迭代
	的最大次数，最后一个是特定的阈值。指定在每个金字塔层，为某点寻找光
	流的迭代过程的终止条件。
  */
  cv::TermCriteria term_criteria;
  float lambda; //某阈值？Lagrangian乘子
  /*
	NCC归一化交叉相关，FB error与NCC结合，使跟踪更稳定，交叉相关的图像
	匹配算法？
	交叉相关法的作用是进行云团移动的短时预测。选取连续两个时次的GMS-5
	卫星云图，将云图区域划分为32*32像素的图像子集，采用交叉相关法计算
	获取两幅云图的最佳匹配区域，根据前后云图匹配区域的位置和时间间隔，
	确定出每个图像子集的移动矢量（速度和方向），并对图像子集的移动矢量
	进行客观分析，其后，基于检验后的图移动矢量集，利用前后轨迹方法对云
	图作短时外推预测。
  */
  void normCrossCorrelation(const cv::Mat& img1,const cv::Mat& img2, std::vector<cv::Point2f>& points1, std::vector<cv::Point2f>& points2);
  bool filterPts(std::vector<cv::Point2f>& points1,std::vector<cv::Point2f>& points2);
public:
  LKTracker();
  bool trackf2f(const cv::Mat& img1, const cv::Mat& img2,
                std::vector<cv::Point2f> &points1, std::vector<cv::Point2f> &points2);
  float getFB(){return fbmed;}
};

