#include "tld_utils.h"
#include <opencv2/opencv.hpp>

//ʹ�ý��������������٣��������Ա�������кܶ���OpenCV��calcOpticalFlowPyLK()
//�����Ĳ���
class LKTracker{
private:
  std::vector<cv::Point2f> pointsFB;
  cv::Size window_size;  //ÿ������������������ڳߴ�
  int level; //���Ľ���������
  std::vector<uchar> status; //�������飬�����Ӧ�����Ĺ��������֣�����
							 //�е�ÿ��Ԫ�ض�������Ϊ1����������Ϊ0
  std::vector<uchar> FB_status;
  std::vector<float> similarity; //���ƶ�
  std::vector<float> FB_error;   //Forward-Backward error��������FB_error
								 //�Ľ����ԭʼλ�õ�ŷʽ�������Ƚϣ�
								 //�Ѿ������ĸ��ٽ������
  float simmed;
  float fbmed; 
  /*
	TermCriteriaģ���࣬ȡ����֮ǰ��CvTermCriteria�����������Ϊ������
	������ֹ�����ģ����������Ҫ3��������һ�������ͣ��ڶ�������Ϊ����
	�������������һ�����ض�����ֵ��ָ����ÿ���������㣬Ϊĳ��Ѱ�ҹ�
	���ĵ������̵���ֹ������
  */
  cv::TermCriteria term_criteria;
  float lambda; //ĳ��ֵ��Lagrangian����
  /*
	NCC��һ��������أ�FB error��NCC��ϣ�ʹ���ٸ��ȶ���������ص�ͼ��
	ƥ���㷨��
	������ط��������ǽ��������ƶ��Ķ�ʱԤ�⡣ѡȡ��������ʱ�ε�GMS-5
	������ͼ������ͼ���򻮷�Ϊ32*32���ص�ͼ���Ӽ������ý�����ط�����
	��ȡ������ͼ�����ƥ�����򣬸���ǰ����ͼƥ�������λ�ú�ʱ������
	ȷ����ÿ��ͼ���Ӽ����ƶ�ʸ�����ٶȺͷ��򣩣�����ͼ���Ӽ����ƶ�ʸ��
	���п͹۷�������󣬻��ڼ�����ͼ�ƶ�ʸ����������ǰ��켣��������
	ͼ����ʱ����Ԥ�⡣
  */
  void normCrossCorrelation(const cv::Mat& img1,const cv::Mat& img2, std::vector<cv::Point2f>& points1, std::vector<cv::Point2f>& points2);
  bool filterPts(std::vector<cv::Point2f>& points1,std::vector<cv::Point2f>& points2);
public:
  LKTracker();
  bool trackf2f(const cv::Mat& img1, const cv::Mat& img2,
                std::vector<cv::Point2f> &points1, std::vector<cv::Point2f> &points2);
  float getFB(){return fbmed;}
};

