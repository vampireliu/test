#include <opencv2/opencv.hpp>
#include <opencv2/legacy/legacy.hpp>
#include "tld_utils.h"
#include "LKTracker.h"
#include "FerNNClassifier.h"
#include <fstream>


//结构体定义，结构体实际上也属于类的范畴
//Bounding Boxes
struct BoundingBox : public cv::Rect   //继承Rect类 
{
  BoundingBox()//类的构造函数
  {
  } 
  BoundingBox(cv::Rect r): cv::Rect(r) //构造函数的重构
  {
  } 
public:
  float overlap;        //Overlap with current Bounding Box
  float koverlap;
  int sidx;             //scale index 
};

//检测 结构体
struct DetStruct 
{
    std::vector<int> bb;
    std::vector<std::vector<int> > patt;
    std::vector<float> conf1;
    std::vector<float> conf2;
    std::vector<std::vector<int> > isin;
    std::vector<cv::Mat> patch;
 };

//Temporal structure
  struct TempStruct 
  {
    std::vector<std::vector<int> > patt;
    std::vector<float> conf;
  };

struct OComparator  //比较两者的重合度
{  
  OComparator(const std::vector<BoundingBox>& _grid):grid(_grid) //构造函数
  {
  }
  std::vector<BoundingBox> grid;
  bool operator()(int idx1,int idx2)
  {
    return grid[idx1].overlap > grid[idx2].overlap;
  }
};
struct CComparator  //比较两者的确信度
{
  CComparator(const std::vector<float>& _conf):conf(_conf)
  {
  }
  std::vector<float> conf;
  bool operator()(int idx1,int idx2)
  {
    return conf[idx1]> conf[idx2];
  }
};


class TLD{
private:
  cv::PatchGenerator generator; //PatchGenerator类用来对图像区域
								//进行仿射变换
  FerNNClassifier classifier;
  LKTracker tracker;

  ///Parameters  
  //下面的参数通过程序开始运行时读入的parameters.yml文件进行了初始化
  int bbox_step;
  int min_win;
  int patch_size;

  //initial parameters for positive examples
  //从第一帧得到的目标的bounding box中（文件读取或者用户框定），经过几何
  //变换得到num_closest_init*num_warps_init个正样本
  int num_closest_init; //最近邻窗口数 （传递参数设为10）
  int num_warps_init;   //几何变换数目(默认20)
  int noise_init;
  float angle_init;
  float shift_init;
  float scale_init;

  //从跟踪得到的目标的bounding box中，经过几何变换更新正样本（添加到在线
  //模型）
  //update parameters for positive examples
  int num_closest_update;
  int num_warps_update;
  int noise_update;
  float angle_update;
  float shift_update;
  float scale_update;
  //parameters for negative examples
  float bad_overlap;
  float bad_patches;
  ///Variables
//Integral Images 积分图像
  //积分图像，用以计算2bitBP特征（类似于haar特征的计算）Mat 最大的优势跟
  //STL很相似，都是对内存进行动态的管理，不需要之前用户手动的管理内存。
  cv::Mat iisum;
  cv::Mat iisqsum;
  float var;
//Training data  训练数据
  //std::pair主要的作用是将两个数据组合成一个数据，两个数据可以是同一类型
  //或者不同类型。pair实质上是一个结构体，其主要成员变量是first和second
  //这两个变量可以直接使用。在这里用来表示样本，first成员为features特征
  //点数组，second成员为labels样本类别标签。

  std::vector<std::pair<std::vector<int>,int> > pX; //正样本positive ferns <features,labels=1>
  std::vector<std::pair<std::vector<int>,int> > nX; //负样本 negative ferns <features,labels=0>
  cv::Mat pEx;  //positive NN example
  std::vector<cv::Mat> nEx; //negative NN examples
//Test data      测试数据
  std::vector<std::pair<std::vector<int>,int> > nXT; //negative data to Test
  std::vector<cv::Mat> nExT; //negative NN examples to Test
//Last frame data
  BoundingBox lastbox;
  bool lastvalid;
  float lastconf;
//Current frame data
  //Tracker data
  bool tracked;
  BoundingBox tbb;
  bool tvalid;
  float tconf;
  //Detector data  检测数据
  TempStruct tmp;
  DetStruct dt;
  std::vector<BoundingBox> dbb;
  std::vector<bool> dvalid;
  std::vector<float> dconf;
  bool detected;
  //Kalman data 
  cv::Mat measurement;
  bool getstatepost = true;
  cv::KalmanFilter KF;
  cv::Rect kalmanbox;
  std::vector<BoundingBox> koverlapbox;//存放与kalmanbox重叠的box
  std::vector<int> koverlapidx;
  //Bounding Boxes
  bool firsttime = true;
  std::vector<BoundingBox> grid;   //用于存储边界方格的容器
  std::vector<cv::Size> scales;    //用于存储尺度，高/宽
  std::vector<int> good_boxes; //indexes of bboxes with overlap > 0.6
  std::vector<int> bad_boxes; //indexes of bboxes with overlap < 0.2
  BoundingBox bbhull; // hull of good_boxes 
					  //good_boxes的壳，也就是窗口的边框
  BoundingBox best_box; // maximum overlapping bbox

public:
  //Constructors
  
  TLD();
  TLD(const cv::FileNode& file);
  void read(const cv::FileNode& file);
  //Methods
  void init(const cv::Mat& frame1,const cv::Rect &box, FILE* bb_file);
  void generatePositiveData(const cv::Mat& frame, int num_warps);
  void generateNegativeData(const cv::Mat& frame);
  void processFrame(const cv::Mat& img1,const cv::Mat& img2,std::vector<cv::Point2f>& points1,std::vector<cv::Point2f>& points2,
      BoundingBox& bbnext,bool& lastboxfound, bool tl,FILE* bb_file);
  void track(const cv::Mat& img1, const cv::Mat& img2,std::vector<cv::Point2f>& points1,std::vector<cv::Point2f>& points2);
  void detect(const cv::Mat& frame);//原始  const  cv::Mat& frame
  void clusterConf(const std::vector<BoundingBox>& dbb,const std::vector<float>& dconf,std::vector<BoundingBox>& cbb,std::vector<float>& cconf);
  void evaluate();
  void learn(const cv::Mat& img);
  //Tools
  void buildGrid(const cv::Mat& img, const cv::Rect& box);
  float bbOverlap(const BoundingBox& box1,const BoundingBox& box2);
  void getOverlappingBoxes(const cv::Rect& box1,int num_closest);
  void getBBHull();
  void getPattern(const cv::Mat& img, cv::Mat& pattern,cv::Scalar& mean,cv::Scalar& stdev);
  void bbPoints(std::vector<cv::Point2f>& points, const BoundingBox& bb);
  void bbPredict(const std::vector<cv::Point2f>& points1,const std::vector<cv::Point2f>& points2,
      const BoundingBox& bb1,BoundingBox& bb2);
  double getVar(const BoundingBox& box,const cv::Mat& sum,const cv::Mat& sqsum);
  bool bbComp(const BoundingBox& bb1,const BoundingBox& bb2);
  int clusterBB(const std::vector<BoundingBox>& dbb,std::vector<int>& indexes);
};

