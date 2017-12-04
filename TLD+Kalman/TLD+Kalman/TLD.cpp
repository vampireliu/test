/*
 * TLD.cpp
 *
 *  Created on: Jun 9, 2011
 *      Author: alantrrs
 */
#include <cstdlib>
#include <sys/timeb.h>

#include "TLD.h"
#include <stdio.h>
#include<fstream>
using namespace cv;
using namespace std;
/*
int round(float f)
{
if ((int)f+0.5>f)
return (int)f;
else
return (int)f + 1;
}
*/

TLD::TLD()
{
}
TLD::TLD(const FileNode& file){
	read(file);
}

void TLD::read(const FileNode& file){
	///Bounding Box Parameters
	min_win = (int)file["min_win"];//15
	///Genarator Parameters
	//initial parameters for positive examples
	patch_size = (int)file["patch_size"];//15
	num_closest_init = (int)file["num_closest_init"];//10
	num_warps_init = (int)file["num_warps_init"];//20
	noise_init = (int)file["noise_init"];//5
	angle_init = (float)file["angle_init"];//20
	shift_init = (float)file["shift_init"];//0.02
	scale_init = (float)file["scale_init"];//0.02
	//update parameters for positive examples
	num_closest_update = (int)file["num_closest_update"];//10
	num_warps_update = (int)file["num_warps_update"];//10
	noise_update = (int)file["noise_update"];//5
	angle_update = (float)file["angle_update"];//10
	shift_update = (float)file["shift_update"];//0.02
	scale_update = (float)file["scale_update"];//0.02
	//parameters for negative examples
	bad_overlap = (float)file["overlap"];//0.2
	bad_patches = (int)file["num_patches"];//100
	classifier.read(file);
}

// TLD��ʼ��
void TLD::init(const Mat& frame1, //��ʾ��Ƶ�ĳ�ʼ֡
	const Rect& box,   //�˹�ѡȡ�ı߽����
	FILE* bb_file      //��¼�߽���ӵ�txt�ļ� �������
	)
{
	//bb_file = fopen("bounding_boxes.txt","w");
	//Get Bounding Boxes
	buildGrid(frame1, box); //�˺������ݴ����box��Ŀ��߽���ڴ���
	//��frame1�й���ȫ����ɨ�贰�ڲ������ص���
	printf("Created %d bounding boxes\n", (int)grid.size());//�˴����ɴ�����box
	///Preparation
	//allocation ����
	//iisum iisqsum���ǻ���ͼ�� ����ͼ�������ڼ��ͼ���ķ����ͼ���ķ���С��һ����ֵ�ͽ���ͼ����ų�
	//Mat�Ĵ�������ʽ�����֣�1������Create(����������������)
	//2��Mat(���������������ͣ�ֵ��)��
	iisum.create(frame1.rows + 1, frame1.cols + 1, CV_32F);  //��������ͼ��
	iisqsum.create(frame1.rows + 1, frame1.cols + 1, CV_64F); //��������ͼ�񣬴�����ͼ���ԭʼͼ���һ��һ�У�Ϊʲô��һ��һ�У�

	/*
	  Detector data�ж��壺std::vector<float> dconf; ���ȷ�Ŷȣ�
	  vector��reserve������vector��capacity����������sizeû�иı䣡
	  ��resize�ı���vector��capacityͬʱҲ����������size!
	  reserve������Ԥ���ռ䣬���ڿռ��ڲ��������Ĵ���Ԫ�ض�������û����
	  ���µĶ���֮ǰ���������������ڵ�Ԫ�ء�
	  �����ǵ���resize���ǵ���reserve�����߶�����ԭ�е�Ԫ�ض�û��Ӱ��
	  ����
	  myVec.reserve(100); //��Ԫ�ػ�û�й��죬��ʱ������[]����Ԫ��
	  myVec.resize(100);  //��Ԫ�ص�Ĭ�Ϲ��캯��������100���µ�Ԫ�أ�
	  //����ֱ�Ӳ�����Ԫ�ء�
	  */
	//������ݣ����洢
	dconf.reserve(100); //װ�ظ������ݵ�����
	dbb.reserve(100);   //װ�ر߽���ӵ�������dbb��bounding box������ 
	bbox_step = 7;       //

	/*
	  ��������Detector data�ж������������������grid.size()��С
	  ������һ��ͼ����ȫ����ɨ�贰�ڸ�����������
	  Detector data�ж���TempStruct tmp;
	  */
	//tmp.conf.reserve(grid.size());
	tmp.conf = vector<float>(grid.size());
	//���������������������ɺ�ռ�ڴ�
	tmp.patt = vector<vector<int> >(grid.size(), vector<int>(10, 0));
	//tmp.patt.reserve(grid.size());
	dt.bb.reserve(grid.size());
	good_boxes.reserve(grid.size());
	bad_boxes.reserve(grid.size());


	//TLD�ж��� cv::Mat pEx; //positive NN example ��СΪ15*15ͼ��Ƭ
	//����Pר��
	pEx.create(patch_size, patch_size, CV_64F);//patch_size=15;

	/*
	  //Init Generator ��ʼ��������
	  //���������࣬�����ų�����Ĳ��Ϲ����ͼ��飬����
	  //TLD�ж��壺cv::PatchGenerator generator;
	  //PatchGenerator����������ͼ��������з���任

	  cv::PatchGenerator::PatchGenerator(
	  double _backgroundMin,
	  double _backgroundMax,
	  double _noiseRange,
	  bool _randomBlur = true,
	  double _lambdaMin = 0.6
	  double _lambdaMax = 1.5,
	  double _thetaMin = -CV_PI,
	  double _thetaMax = CV_PI,
	  double _phiMin = -CV_PI,
	  double _phiMax = CV_PI
	  )
	  һ����÷����ȳ�ʼ��һ��PatchGeneratorʵ����Ȼ��RNGһ���������
	  �ٵ��ã������������һ���任�����������
	  */
	generator = PatchGenerator(0, 0, noise_init, true,
		1 - scale_init,
		1 + scale_init,
		-angle_init*CV_PI / 180,
		angle_init*CV_PI / 180,
		-angle_init*CV_PI / 180,
		angle_init*CV_PI / 180);
	/*
	  ����ĺ������ݴ����box��Ŀ��߽�򣩣�����֡ͼ���е�ȫ��ɨ�贰���У���buildGrid���ɣ�Ѱ�����box������С
	  ���������ƣ��ص�����󣩵�num_closest_init(10)�����ڣ�Ȼ�����Щ���ڹ�Ϊgood_boxes������ͬʱ���ص���
	  С��0.2�Ĺ��뵽bad_boxes�������൱�ڶ�ȫ����ɨ�贰�ڽ���ɸѡ����ͨ��BBhull�����õ���Щɨ�贰�ڵ����߽硣

	  �����Ǹ���overlap�ı�����Ϣѡ���ظ������������60%����num_closest_init=10��
	  ����ӽ�box��RectBox,�൱�ڶ�RectBox����ɸѡ����ͨ��BBhull�����õ���ЩRectBox
	  �����߽硣
	  */
	getOverlappingBoxes(box, num_closest_init);
	printf("Found %d good boxes, %d bad boxes\n", (int)good_boxes.size(), (int)bad_boxes.size());
	printf("Best Box: %d %d %d %d\n", best_box.x, best_box.y, best_box.width, best_box.height);
	printf("Bounding box hull: %d %d %d %d\n", bbhull.x, bbhull.y, bbhull.width, bbhull.height);
	//Correct Bounding Box
	lastbox = best_box;
	lastconf = 1;
	lastvalid = true;
	//Print
	fprintf(bb_file, "%d,%d,%d,%d,%f\n", lastbox.x, lastbox.y, lastbox.br().x, lastbox.br().y, lastconf);//����߽�box��txt�ļ�
	//Prepare Classifier ׼��������
	//scales������������ɨ�贰�ڵĳ߶ȣ���buildGrid()������ʼ��
	classifier.prepare(scales);//prepare�Ĺ������Ǹ�ÿһ��ɨ�贰�ڳ�ʼ����Ӧ��pixel comparisons��������������
	///Generate Data                  //���ص����꣩��Ȼ���ʼ���������Ϊ0��
	// Generate positive data
	generatePositiveData(frame1, num_warps_init);
	// Set variance threshold
	Scalar stdev, mean;
	/*
	  ͳ��best_box�ľ�ֵ�ͱ�׼��
	  ������Ҫ��ȡͼ��A��ĳ��ROI�Ļ�����Mat���B=img(ROI)������ȡ
	  frame1(best_box) �ͱ�ʾ��frame1����ȡbest_box����Ŀ�����򣩵�ͼ��Ƭ
	  */
	meanStdDev(frame1(best_box), mean, stdev);

	/*
	  ���û���ͼ��ȥ����ÿ�����ⴰ�ڵķ���
	  cvIntegral(const CvArr* image, CvArr* sum, CvArr* sqsum=NULL,CvArr*
	  tilted_sum=NULL);
	  �������ͼ��image ����ͼ��sum ����ͼ��W+1xH+1�� sqsum ������ֵƽ���Ļ���ͼ��
	  tilted_sum ��ת45�Ļ���ͼ��
	  ���û���ͼ�񣬿��Լ�����ĳ���ص���-�ҷ��Ļ�����ת�ľ��������н�����͡����ֵ�Լ���׼����ļ���
	  ���ұ�֤���㸴�Ӷ�ΪO(1)
	  */
	integral(frame1, iisum, iisqsum);//iisum����ͼ��iisqsum������ֵƽ���Ļ���ͼ

	/*
	  ����������ģ��һ��������ģ�飬���û���ͼ����ÿ������ⴰ�ڵķ���������var��ֵ
	  ��Ŀ��patch�����50%���ġ�����Ϊ�京��ǰ��Ŀ��
	  ����var��Ϊ��׼���ƽ����
	  */
	var = pow(stdev.val[0], 2)*0.5; //getVar(best_box,iisum,iisqsum);
	cout << "variance: " << var << endl;
	//check variance
	//getVar����ͨ������ͼ����������best_box�ķ���
	double vr = getVar(best_box, iisum, iisqsum)*0.5;
	cout << "check variance: " << vr << endl;
	// Generate negative data ����������
	generateNegativeData(frame1);
	//Split Negative Ferns into Training and Testing sets (they are already shuffled)
	//���������Ž�ѵ���Ͳ��Լ�
	int half = (int)nX.size()*0.5f;
	//��һ��ĸ���������Ϊ���Լ���Ȼ��ʣ�µ�һ����Ϊѵ����
	nXT.assign(nX.begin() + half, nX.end()); //nXT��negative data to Test
	nX.resize(half);

	///Split Negative NN Examples into Training and Testing sets
	half = (int)nEx.size()*0.5f;
	nExT.assign(nEx.begin() + half, nEx.end());
	nEx.resize(half);

	//Merge Negative Data with Positive Data and shuffle it
	//�����������������ϲ���Ȼ�����
	vector<pair<vector<int>, int> > ferns_data(nX.size() + pX.size());
	vector<int> idx = index_shuffle(0, ferns_data.size());
	int a = 0;
	for (int i = 0; i < pX.size(); i++){
		ferns_data[idx[a]] = pX[i];
		a++;
	}
	for (int i = 0; i < nX.size(); i++){
		ferns_data[idx[a]] = nX[i];
		a++;
	}
	//Data already have been shuffled, just putting it in the same vector
	vector<cv::Mat> nn_data(nEx.size() + 1);//nn_data��������ڷ�������ѵ��
	nn_data[0] = pEx;
	for (int i = 0; i < nEx.size(); i++){
		nn_data[i + 1] = nEx[i];
	}

	///Training
	//ѵ�����Ϸ�������ɭ�֣�������ڷ�����
	classifier.trainF(ferns_data, 2); //bootstrap = 2
	classifier.trainNN(nn_data);
	///Threshold Evaluation on testing sets
	//������������õ��ļ��Ϸ�������ɭ�֣�������ڷ������н��з��࣬����
	//�õ�����ֵ
	classifier.evaluateTh(nXT, nExT);
}//�˴���ʼ������

/* Generate Positive data
 * Inputs:
 * - good_boxes (bbP)
 * - best_box (bbP0)
 * - frame (im0)
 * Outputs:
 * - Positive fern features (pX)
 * - Positive NN examples (pEx)
 */
void TLD::generatePositiveData(const Mat& frame, int num_warps)
{
	/*
	  CvScalar����ɴ��1-4����ֵ�����飬��ṹ�����£�
	  typedef struct CvScalar
	  {
	  double val[4];
	  }CvScalar;
	  ���ʹ�õ�ͼ����1ͨ���ģ���s.val[0]�д洢����
	  ���ʹ�õ�ͼ����3ͨ���ģ���s.val[0]��s.val[1]��s.val[2]�д洢����
	  */
	Scalar mean; //��ֵ
	Scalar stdev;//��׼��

	getPattern(frame(best_box), pEx, mean, stdev);//�˺�����frameͼ��best_box
	//�����ͼ��Ƭ��һ��Ϊ��ֵ
	//Ϊ0��15*15��patch���洢��
	//pEx����������ڷ���������������������
	//Get Fern features on warped patches
	Mat img;//img��˹����������֡
	Mat warped;
	GaussianBlur(frame, img, Size(9, 9), 1.5);//��9*9�ĸ�˹��ģ������֡

	/*
	  ��imgͼ���н�ȡbbhull��Ϣ��bbhull�ǰ�����λ�úʹ�С�ľ��ο򣩵�ͼ��
	  ����warped��������Ҫ��ȡͼ��A��ĳ��ROI�Ļ�����B=img(ROI)������ȡ
	  */
	warped = img(bbhull);
	RNG& rng = theRNG(); //����һ�������
	//ȡ���δ����ĵ�����int i(2)
	Point2f pt(bbhull.x + (bbhull.width - 1)*0.5f, bbhull.y + (bbhull.height - 1)*0.5f);

	/*
	  nstructs��Ŀ����һ�������鹹����ÿ����������ͼ���Ĳ�ͬ��ͼ��ʾ��
	  �ĸ��� fern[nstructs] nstructs������ɭ�ֵ����飿
	  */
	vector<int> fern(classifier.getNumStructs());
	pX.clear();
	Mat patch;

	//pXΪ������RectBox���߽紦����������Ϣ��pEx����ڵ�RectBox��
	//Pattern��bbP0Ϊ����ڵ�RectBox
	if (pX.capacity() < num_warps*good_boxes.size())
		pX.reserve(num_warps*good_boxes.size()); //pX����������Ϊ����任
	//����*goog_box�ĸ�������
	//��������������ô��Ŀռ�
	int idx;
	for (int i = 0; i<num_warps; i++)
	{
		if (i>0)
			generator(frame, pt, warped, bbhull.size(), rng); //PatchGenerator������
		//��ͼ��������з���任��
		//��RNGһ��������ӣ��ٵ�
		//�ã������������һ���任
		//����������� 
		for (int b = 0; b < good_boxes.size(); b++)
		{
			idx = good_boxes[b];//good_boxes����������� grid ������  
			patch = img(grid[idx]); //��img�� grid[idx] ����Ҳ����bounding box��
			//���ȸߵģ���һ��ͼ��Ƭ��ȡ����   patchΪ�ص��ȸߵ�����
			classifier.getFeatures(patch, grid[idx].sidx, fern);//getFeatures�����õ�
			//�����patch��������
			//�Ľڵ㣬Ҳ����������
			//������fern��13λ�Ķ�
			//���ƴ��룩
			//grid[idx].sidx ��Ӧ��
			//�߶����� 
			pX.push_back(make_pair(fern, 1)); //positive ferns <features, labels=1>  ���Ϊ��
			//����     make_pair��������Ԫ��ı�������ģ��
		}                                    //make_pair����������������Ϊ��������Ľṹ
	}
	printf("Positive examples generated: ferns:%d NN:1\n", (int)pX.size());//����200��
}

/*
	�ȶ���ӽ�box��RectBox����õ���patch ,Ȼ��������Ϣת��ΪPattern,�����˵����
	��һ��RectBox��Ӧ��patch��size��������patch_size = 15*15������2ά�ľ�����һά
	��������Ϣ,Ȼ��������Ϣ��ֵ��Ϊ0������Ϊzero mean and unit variance��ZMUV��
	Output: resized Zero-Mean patch
	*/
void TLD::getPattern(const Mat& img, Mat& pattern, Scalar& mean, Scalar& stdev){
	//Output: resized Zero-Mean patch
	resize(img, pattern, Size(patch_size, patch_size));//��img������patch_size = 15*15��
	//�浽pattern�� 
	meanStdDev(pattern, mean, stdev);  //����pattern�������ľ�ֵ�ͱ�׼�� 
	pattern.convertTo(pattern, CV_32F);
	/*
	  opencv��Mat������������أ� Mat���� + Mat; + Scalar; + int / float / double ������
	  ����������Ԫ�ؼ�ȥ���ֵ��Ҳ���ǰ�patch�ľ�ֵ��Ϊ��
	  */
	pattern = pattern - mean.val[0];
}

/*TLD::generateNegativeData(const Mat& frame)
 * Inputs:
 * - Image
 * - bad_boxes (Boxes far from the bounding box)
 * - variance (pEx variance)
 * Outputs
 * - Negative fern features (nX)
 * - Negative NN examples (nEx)
 */
void TLD::generateNegativeData(const Mat& frame)
{
	/*
	  ����֮ǰ�ص���С��0.2�ģ������� bad_boxes�ˣ���������ͦ�࣬����ĺ������ڴ���˳
	  ��Ҳ����Ϊ�˺������ѡ��bad_boxes
	  */
	random_shuffle(bad_boxes.begin(), bad_boxes.end());//Random shuffle bad_boxes indexes
	int idx;
	//Get Fern Features of the boxes with big variance (calculated using integral images)
	int a = 0;
	//int num = std::min((int)bad_boxes.size(),(int)bad_patches*100); //limits the size of bad_boxes to try
	printf("negative data generation started.\n");
	vector<int> fern(classifier.getNumStructs());//return 10
	nX.reserve(bad_boxes.size());
	Mat patch;
	for (int j = 0; j < bad_boxes.size(); j++)
	{
		idx = bad_boxes[j];
		if (getVar(grid[idx], iisum, iisqsum) < var*0.5f)
			continue;
		patch = frame(grid[idx]);
		classifier.getFeatures(patch, grid[idx].sidx, fern);
		nX.push_back(make_pair(fern, 0));
		a++;
	}
	printf("Negative examples generated: ferns: %d ", a);
	//random_shuffle(bad_boxes.begin(),bad_boxes.begin()+bad_patches);
	//Randomly selects 'bad_patches' and get the patterns for NN;
	Scalar dum1, dum2;

	nEx = vector<Mat>(bad_patches); //bad_patches = (int)file["num_patches"]; �ڲ�����
	//���� num_patches = 100  
	for (int i = 0; i < bad_patches; i++){
		idx = bad_boxes[i];
		patch = frame(grid[idx]);
		/*
		  �����˵���ǹ�һ��RectBox��Ӧ��patch��size��������patch_size = 15*15��
		  ���ڸ���������Ҫ��ֵ�ͷ�����ԾͶ���dum����������
		  */
		getPattern(patch, nEx[i], dum1, dum2);
	}
	printf("NN: %d\n", (int)nEx.size());
}

//�ú���ͨ������ͼ����������box�ķ���
double TLD::getVar(const BoundingBox& box, const Mat& sum, const Mat& sqsum)
{
	double brs = sum.at<int>(box.y + box.height, box.x + box.width);
	double bls = sum.at<int>(box.y + box.height, box.x);
	double trs = sum.at<int>(box.y, box.x + box.width);
	double tls = sum.at<int>(box.y, box.x);
	double brsq = sqsum.at<double>(box.y + box.height, box.x + box.width);
	double blsq = sqsum.at<double>(box.y + box.height, box.x);
	double trsq = sqsum.at<double>(box.y, box.x + box.width);
	double tlsq = sqsum.at<double>(box.y, box.x);
	double mean = (brs + tls - trs - bls) / ((double)box.area());
	double sqmean = (brsq + tlsq - trsq - blsq) / ((double)box.area());
	return sqmean - mean*mean;  //����=E(X^2)-(EX)^2   EX��ʾ��ֵ  
}

void TLD::processFrame(const cv::Mat& img1,
	const cv::Mat& img2,
	vector<Point2f>& points1,
	vector<Point2f>& points2,
	BoundingBox& bbnext,
	bool& lastboxfound,
	bool tl, FILE* bb_file)
{
	vector<BoundingBox> cbb;
	vector<float> cconf;
	int confident_detections = 0;
	int didx; //detection index
	//������ģ��ʱ��
	ofstream detecttime;
	detecttime.open("detect_time.txt",ios::app);

	///Track ����ģ��
	if (lastboxfound && tl)
	{
		
		track(img1, img2, points1, points2); //���� img2Ϊ��ǰ֡,img1��һ֡�Ҷ�ͼ

		}
	else
	{
		tracked = false;
	}
	///Detect ���ģ��
	struct timeb start, end;
	ftime(&start);

	detect(img2);//img2Ϊ��ǰ֡

	ftime(&end);
	std::cout << "2 .detect Process Time: " << (end.time - start.time) * 1000 + (end.millitm - start.millitm) << " ms" << endl;
	detecttime << "detecttimeframe Process Time: " << (end.time - start.time) * 1000 + (end.millitm - start.millitm) << "ms" << endl;
	///Integration �ۺ�ģ��


	if (tracked)//TLDֻ���ٵ�Ŀ�꣬�����ۺ�ģ���ۺϸ��������ٵ��ĵ���Ŀ��ͼ������⵽�Ķ��Ŀ��
		//��Ȼ��ֻ����������ƶ�����һ��Ŀ��
	{
		bbnext = tbb;
		lastconf = tconf; //��ʾ������ƶȵ���ֵ
		lastvalid = tvalid; //��ʾ�������ƶȵ���ֵ 
		printf("Tracked\n");
		if (detected){                                               //   if Detected
			//ͨ�� �ص��� �Լ������⵽��Ŀ��bounding box���о��࣬ÿ�������ص���С
			//��0.5 
			clusterConf(dbb, dconf, cbb, cconf);                       //   cluster detections
			printf("Found %d clusters\n", (int)cbb.size());
			for (int i = 0; i < cbb.size(); i++)
			{
				//�ҵ�����������ٵ���box����Ƚ�Զ���ࣨ�������⵽��box������������
				//������ƶȱȸ�������Ҫ��  
				if (bbOverlap(tbb, cbb[i])<0.5 && cconf[i]>tconf){  //  Get index of a clusters that is far from tracker and are more confident than the tracker
					confident_detections++;  //��¼��������������Ҳ���ǿ��ŶȱȽϸߵ�Ŀ��box�ĸ���
					didx = i; //detection index
				}
			}
			if (confident_detections == 1)//if there is ONE such a cluster, re-initialize the tracker
			{   //���ֻ��һ����������������box����ô�������Ŀ��box�����³�ʼ��������
				//��Ҳ�����ü�����Ľ��ȥ������������                             
				printf("Found a better match..reinitializing tracking\n");
				bbnext = cbb[didx];
				lastconf = cconf[didx];
				lastvalid = false;
			}
			else
			{
				printf("%d confident cluster was found\n", confident_detections);
				int cx = 0, cy = 0, cw = 0, ch = 0;
				int close_detections = 0;
				for (int i = 0; i<dbb.size(); i++)
				{
					//�ҵ��������⵽��box�������Ԥ�⵽��box����ܽ����ص��ȴ���0.7
					//����box����������ʹ�С�����ۼ�  
					if (bbOverlap(tbb, dbb[i])>0.7)// Get mean of close detections
					{
						cx += dbb[i].x;
						cy += dbb[i].y;
						cw += dbb[i].width;
						ch += dbb[i].height;
						close_detections++; //��¼�����box�ĸ��� 
						printf("weighted detection: %d %d %d %d\n", dbb[i].x, dbb[i].y, dbb[i].width, dbb[i].height);
					}
				}
				//���������Ԥ�⵽��box����ܽ���box �� ����������Ԥ�⵽��box ������
				//�����С��ƽ����Ϊ���յ�Ŀ��bounding box�����Ǹ�������Ȩֵ�ϴ�  
				if (close_detections > 0)
				{
					bbnext.x = cvRound((float)(10 * tbb.x + cx) / (float)(10 + close_detections));   // weighted average trackers trajectory with the close detections
					bbnext.y = cvRound((float)(10 * tbb.y + cy) / (float)(10 + close_detections));
					bbnext.width = cvRound((float)(10 * tbb.width + cw) / (float)(10 + close_detections));
					bbnext.height = cvRound((float)(10 * tbb.height + ch) / (float)(10 + close_detections));
					printf("Tracker bb: %d %d %d %d\n", tbb.x, tbb.y, tbb.width, tbb.height);
					printf("Average bb: %d %d %d %d\n", bbnext.x, bbnext.y, bbnext.width, bbnext.height);
					printf("Weighting %d close detection(s) with tracker..\n", close_detections);
				}
				else{
					printf("%d close detections were found\n", close_detections);

				}
			}
		}
	}
	else
	{                                       //   If NOT tracking
		printf("Not tracking..\n");
		lastboxfound = false;
		lastvalid = false;
		//���������û�и��ٵ�Ŀ�꣬���Ǽ������⵽��һЩ���ܵ�Ŀ��box����ôͬ������
		//���о��࣬��ֻ�Ǽ򵥵Ľ������cbb[0]��Ϊ�µĸ���Ŀ��box ���³�ʼ��������   
		if (detected){                           //  and detector is defined
			clusterConf(dbb, dconf, cbb, cconf);   //  cluster detections
			printf("Found %d clusters\n", (int)cbb.size());
			if (cconf.size() == 1){
				bbnext = cbb[0];
				lastconf = cconf[0];
				printf("Confident detection..reinitializing tracker\n");
				lastboxfound = true;
			}
		}
	}
	lastbox = bbnext;
	if (lastboxfound)
		fprintf(bb_file, "%d,%d,%d,%d,%f\n", lastbox.x, lastbox.y, lastbox.br().x, lastbox.br().y, lastconf);
	else
		fprintf(bb_file, "NaN,NaN,NaN,NaN,NaN\n");

	///learn ѧϰģ��
	if (lastvalid && tl)
		learn(img2);
}

/*Inputs:
   * -current frame(img2), last frame(img1), last Bbox(bbox_f[0]).
   *Outputs:
   *- Confidence(tconf), Predicted bounding box(tbb),Validity(tvalid), points2 (for display purposes only)
   */
void TLD::track(const Mat& img1,
	const Mat& img2,
	vector<Point2f>& points1,
	vector<Point2f>& points2)
{

	//Generate points
	//����������㣨���Ȳ���������lastbox�й��������10*10=100�������㣬����points1  
	bbPoints(points1, lastbox);
	if (points1.size() < 1)
	{
		printf("BB= %d %d %d %d, Points not generated\n", lastbox.x, lastbox.y, lastbox.width, lastbox.height);
		tvalid = false;
		tracked = false;
		return;
	}
	vector<Point2f> points = points1;
	//Frame-to-frame tracking with forward-backward error cheking
	/*
	  trackf2f������ɣ����١�����FB error��ƥ�����ƶ�sim��Ȼ��ɸѡ��
	  FB_error[i] <= median(FB_error) ��sim_error[i] > median(sim_error) ��������
	  �����ٽ�����õ������㣩��ʣ�µ��ǲ���50%��������
	  */
	tracked = tracker.trackf2f(img1, img2, points, points2);
	if (tracked)
	{
		//Bounding box prediction
		bbPredict(points, points2, lastbox, tbb);//����ʣ�µ��ⲻ��һ��ĸ��ٵ�������Ԥ��
		//Ŀ����ڵ�ǰ֡��λ�úʹ�С tbb  

		/*
		  ����ʧ�ܼ�⣺���FB error����ֵ����10�����أ�����ֵ��������Ԥ�⵽�ĵ�ǰbox��λ
		  ���Ƴ�ͼ������Ϊ���ٴ��󣬴�ʱ������bounding box��Rect::br()���ص������½�
		  ������getFB()���ص���FB error����ֵ
		  */
		if (tracker.getFB() > 10 || tbb.x > img2.cols || tbb.y > img2.rows || tbb.br().x < 1 || tbb.br().y < 1)
		{
			tvalid = false; //too unstable prediction or bounding box out of image
			tracked = false;
			printf("Too unstable predictions FB error=%f\n", tracker.getFB());
			return;
		}
		//Estimate Confidence and Validity //��������ȷ�ŶȺ���Ч��
		Mat pattern;
		Scalar mean, stdev;
		BoundingBox bb;
		bb.x = max(tbb.x, 0);
		bb.y = max(tbb.y, 0);
		bb.width = min(min(img2.cols - tbb.x, tbb.width), min(tbb.width, tbb.br().x));
		bb.height = min(min(img2.rows - tbb.y, tbb.height), min(tbb.height, tbb.br().y));
		getPattern(img2(bb), pattern, mean, stdev);//��һ��img2(bb)��Ӧ��patch��size������
		//��patch_size = 15*15��������pattern  
		vector<int> isin;
		float dummy;
		classifier.NNConf(pattern, isin, dummy, tconf); //Conservative Similarity 
		//����ͼ��Ƭpattern������ģ��M�ı�
		//�����ƶ�
		tvalid = lastvalid;
		if (tconf > classifier.thr_nn_valid)//�������ƶȴ�����ֵ��������������Ч 
		{
			tvalid = true;
		}
	}
	else
		printf("No points tracked\n");

}

//����������㣬box��10*10=100��������  
void TLD::bbPoints(vector<cv::Point2f>& points, const BoundingBox& bb)
{
	int max_pts = 10;
	int margin_h = 0; //�����߽�  
	int margin_v = 0;
	int stepx = ceil(double((bb.width - 2 * margin_h) / max_pts));  // ceil�������ش��ڻ��ߵ���ָ�����ʽ����С����
	int stepy = ceil(double((bb.height - 2 * margin_v) / max_pts));

	//����������㣬box��10*10=100�������� 
	for (int y = bb.y + margin_v; y < bb.y + bb.height - margin_v; y += stepy){
		for (int x = bb.x + margin_h; x < bb.x + bb.width - margin_h; x += stepx){
			points.push_back(Point2f(x, y));
		}
	}
}

//����ʣ�µ��ⲻ��һ��ĸ��ٵ�������Ԥ��bounding box�ڵ�ǰ֡��λ�úʹ�С  
void TLD::bbPredict(const vector<cv::Point2f>& points1,
	const vector<cv::Point2f>& points2,
	const BoundingBox& bb1,
	BoundingBox&       bb2)
{
	int npoints = (int)points1.size();
	vector<float> xoff(npoints); //λ�� 
	vector<float> yoff(npoints);
	printf("tracked points : %d\n", npoints);
	for (int i = 0; i < npoints; i++) //����ÿ������������֮֡���λ��
	{
		xoff[i] = points2[i].x - points1[i].x;
		yoff[i] = points2[i].y - points1[i].y;
	}
	float dx = median(xoff);  //����xλ�Ƶ���ֵ  
	float dy = median(yoff);  //����yλ�Ƶ���ֵ
	float s;
	/*
	  ����bounding box�߶�scale�ı仯��ͨ�����㵱ǰ�������໥��ľ�������ǰ����һ֡��
	  �������໥��ľ���ı�ֵ���Ա�ֵ����ֵ��Ϊ�߶ȵı仯����
	  */
	if (npoints > 1){
		vector<float> d;
		d.reserve(npoints*(npoints - 1) / 2); //�Ȳ�������ͣ�1+2+...+(npoints-1)
		for (int i = 0; i < npoints; i++){
			for (int j = i + 1; j < npoints; j++)//���� ��ǰ�������໥��ľ��� �� ��ǰ����һ֡��
				//�������໥��ľ��� �ı�ֵ��λ���þ���ֵ��
			{
				d.push_back(norm(points2[i] - points2[j]) / norm(points1[i] - points1[j]));
			}
		}
		s = median(d);
	}
	else {
		s = 1.0;
	}
	float s1 = 0.5*(s - 1)*bb1.width;
	float s2 = 0.5*(s - 1)*bb1.height;
	printf("s= %f s1= %f s2= %f \n", s, s1, s2);
	/*
	  �õ���ǰbounding box��λ�����С��Ϣ��ǰbox��x���� = ǰһ֡box��x���� + ȫ��������
	  λ�Ƶ���ֵ�������Ϊbox�ƶ����Ƶ�λ�ƣ� - ��ǰbox���һ��
	  */
	bb2.x = round(bb1.x + dx - s1);
	bb2.y = round(bb1.y + dy - s2);
	bb2.width = round(bb1.width*s);
	bb2.height = round(bb1.height*s);
	printf("predicted bb: %d %d %d %d\n", bb2.x, bb2.y, bb2.br().x, bb2.br().y);
}

void TLD::detect(const cv::Mat& frame)
{
	//cleaning
	dbb.clear();//���Ŀ�������
	dconf.clear();//���Ŀ������Ŷ�
	dt.bb.clear();//���Ŀ�������
	koverlapbox.clear();//������box����
	koverlapidx.clear();
	double t = (double)getTickCount();//GetTickCount���شӲ���ϵͳ������������������ʱ�� 
	Mat img(frame.rows, frame.cols, CV_8U);
	integral(frame, iisum, iisqsum); //����frame�Ļ���ͼ 
	GaussianBlur(frame, img, Size(9, 9), 1.5); //��˹ģ����ȥ�룿
	int numtrees = classifier.getNumStructs();
	float fern_th = classifier.getFernTh();  //getFernTh()����thr_fern; ���Ϸ������ķ�����ֵ  
	vector <int> ferns(10);
	float conf;
	int a = 0;
	Mat patch;
	Point  rectangle_point;
	rectangle_point = Point((tbb.x + tbb.width / 2), (tbb.y + tbb.height / 2));	
	//������ٳɹ�������뿨�����˲�ģ��
	//�������˲�ģ��		  
	if (tracked)
	{
		// KalmanFilter KF;
		

		if (getstatepost == true)
		{

			KF = KalmanFilter(4, 2, 0);
					
			KF.transitionMatrix = *(Mat_<float>(4, 4) <<
				1, 0, 1, 0,
				0, 1, 0, 1,
				0, 0, 1, 0,
				0, 0, 0, 1);  //ת�ƾ���A
			KF.measurementMatrix = *(Mat_<float>(2, 4) <<
				1, 0, 0, 0,
				0, 1, 0, 0
				);                                             //�۲����H
			KF.processNoiseCov = *(Mat_<float>(4, 4) <<		//ϵͳ�����������Q
				1, 0, 0, 0,
				0, 1, 0, 0,
				0, 0, 1, 0,
				0, 0, 0, 1
				);
			KF.measurementNoiseCov = *(Mat_<float>(2, 2) <<		//���������������R
				1, 0,
				0, 1
				);
			KF.errorCovPost*(Mat_<float>(4, 4) <<
				1, 0, 0, 0,
				0, 1, 0, 0,
				0, 0, 1, 0,
				0, 0, 0, 1
				);
			measurement = Mat::zeros(2, 1, CV_32F);

			KF.statePost = *(Mat_<float>(4, 1) <<//statePostΪУ��״̬���䱾�ʾ���ǰһʱ�̵�״̬
				rectangle_point.x,
				rectangle_point.y,
				0,
				0);

			getstatepost = false;
		}
		//2.kalman prediction
		Mat prediction = KF.predict();
		Point predict_pt = Point(prediction.at<float>(0), prediction.at<float>(1));   //Ԥ��ֵ(x',y')
		//3.update measurement
		measurement.at<float>(0) = (float)rectangle_point.x;//������ʵ��λ�ø��£������˺ܾõ����⣡������
		measurement.at<float>(1) = (float)rectangle_point.y;
		//4.update
		KF.correct(measurement);
		//���㿨����Ԥ��box����tbb��3��
		// cv::Rect kalmanbox;
		kalmanbox.x = (predict_pt.x - (1 * tbb.width) / 2)-3;//3��
		kalmanbox.y = (predict_pt.y - (1 * tbb.height) / 2)-3;
		kalmanbox.width = (1 * tbb.width)+3;
		kalmanbox.height = (1 * tbb.height)+3;

		
			
		drawBox(img, kalmanbox);
		
		imshow("kalman prediction", img);

		//��tbbΪ����
/*		kalmanbox.x = rectangle_point.x - (3 * tbb.width) / 2;//3��
		kalmanbox.y = rectangle_point.y - (3 * tbb.height) / 2;
		kalmanbox.width = 3 * tbb.width;
		kalmanbox.height = 3 * tbb.height;*/


		
		int countkalmanbox=0;
		for (int i = 0; i<grid.size(); i++)
		{  //�������е�ɨ�贰����Ŀ��kalmanbox���ص���  
			grid[i].koverlap = bbOverlap(kalmanbox, grid[i]);//����ͨ������������box���ٵ���ֵ
			if (grid[i].koverlap>0.2)
			{
				koverlapidx.push_back(i);
			}	
			//{ 
				//countkalmanbox++;
				//koverlap.push_back(grid[i]);//koverlapidx���kalmanbox������
				//grid.push_back(grid[i]); //û������	��μ�¼�����ţ�		
			//}
		}
		/* for (int i = 0; i < grid.size(); i++)//����������ѭ����Ϊһ��
		{
			if (grid[i].koverlap>0.1)
			{
				koverlapidx.push_back(i);
			}
		}*/


		/*
		  ����������ģ��һ��������ģ�飬���û���ͼ����ÿ������ⴰ�ڵķ���������var
		  ��ֵ��Ŀ��patch�����50%���ģ�����Ϊ�京��ǰ��Ŀ��
		  */
		for (int i = 0; i < koverlapidx.size(); i++){//FIXME: BottleNeck ƿ��  
			if (getVar(grid[koverlapidx[i]], iisum, iisqsum) >= var) //����ÿһ��ɨ�贰�ڵķ���
			{
				a++;
				//����������ģ��������Ϸ��������ģ�� 
				patch = img(grid[koverlapidx[i]]);
				classifier.getFeatures(patch, grid[koverlapidx[i]].sidx, ferns);//�õ���patch������13λ�Ķ�
				//���ƴ��룩  
				conf = classifier.measure_forest(ferns); //���������ֵ��Ӧ�ĺ�������ۼ�ֵ 
				tmp.conf[koverlapidx[i]] = conf; //Detector data�ж���TempStruct tmp; 
				tmp.patt[koverlapidx[i]] = ferns;

				//������Ϸ������ĺ�����ʵ�ƽ��ֵ������ֵfern_th����ѵ���õ�)������Ϊ����ǰ��Ŀ��
				if (conf > numtrees*fern_th)//numtrees=10 fern_th0.6
				{
					dt.bb.push_back(koverlapidx[i]); //��ͨ�������������ģ���ɨ�贰�ڼ�¼��detect structure��
				}
			}
			else{
				tmp.conf[i] = 0.0;
			}
		}
		int detections = dt.bb.size();
		printf("%d Bounding boxes passed the variance filter\n", a);
		printf("%d Initial detection from Fern Classifier\n", detections);
		//���ͨ�������������ģ���ɨ�贰��������100������ֻȡ������ʴ��ǰ100�� 
		if (detections > 100){ //CComparator(tmp.conf)ָ���ȽϷ�ʽ������  
			nth_element(dt.bb.begin(), dt.bb.begin() + 100, dt.bb.end(), CComparator(tmp.conf));
			dt.bb.resize(100);
			detections = 100;
		}
//����kalman box
		  for (int i=0;i<detections;i++){
			  drawBox(img, grid[dt.bb[i]]);
		    }
		  imshow("kalman detections",img);


		if (detections == 0){
			detected = false;
			return;
		}
		printf("Fern detector made %d detections ", detections);
		t = (double)getTickCount() - t; //����ʹ��getTickCount()��Ȼ���ٳ���getTickFrequency()��
		//���������������sΪ��λ��ʱ��
		printf("in %gms\n", t * 1000 / getTickFrequency());
		//  Initialize detection structure
		dt.patt = vector<vector<int> >(detections, vector<int>(10, 0));        //  Corresponding codes of the Ensemble Classifier
		dt.conf1 = vector<float>(detections);                                //  Relative Similarity (for final nearest neighbour classifier)
		dt.conf2 = vector<float>(detections);                                 //  Conservative Similarity (for integration with tracker)
		dt.isin = vector<vector<int> >(detections, vector<int>(3, -1));        //  Detected (isin=1) or rejected (isin=0) by nearest neighbour classifier
		dt.patch = vector<Mat>(detections, Mat(patch_size, patch_size, CV_32F));//  Corresponding patches
		int idx;
		Scalar mean, stdev;
		float nn_th = classifier.getNNTh();


		//����������ģ����������ڷ��������ģ��
		for (int i = 0; i < detections; i++)

		{                                         //  for every remaining detection
			idx = dt.bb[i];                                                       //  Get the detected bounding box index
			patch = frame(grid[idx]);
			getPattern(patch, dt.patch[i], mean, stdev);                //  Get pattern within bounding box
			//����ͼ��Ƭpattern������ģ��M��������ƶȺͱ������ƶ�
			classifier.NNConf(dt.patch[i], dt.isin[i], dt.conf1[i], dt.conf2[i]);  //  Evaluate nearest neighbour classifier
			dt.patt[i] = tmp.patt[idx];
			//printf("Testing feature %d, conf:%f isin:(%d|%d|%d)\n",i,dt.conf1[i],dt.isin[i][0],dt.isin[i][1],dt.isin[i][2]);
			//������ƶȴ�����ֵ������Ϊ����ǰ��Ŀ��
			if (dt.conf1[i] > nn_th)
			{                                               //  idx = dt.conf1 > tld.model.thr_nn; % get all indexes that made it through the nearest neighbour
				dbb.push_back(grid[idx]);                                         //  BB    = dt.bb(:,idx); % bounding boxes
				dconf.push_back(dt.conf2[i]);                                     //  Conf  = dt.conf2(:,idx); % conservative confidences
			}
		}

		//  end
		if (dbb.size() > 0){ //��ӡ��⵽�Ŀ��ܴ���Ŀ���ɨ�贰����������ͨ����������������ģ�
			printf("Found %d NN matches\n", (int)dbb.size());
			detected = true;
		}
		else{
			printf("No NN matches found.\n");
			detected = false;
		}
	}
	/*�������ʧ��
	����������ģ��һ��������ģ�飬���û���ͼ����ÿ������ⴰ�ڵķ���������var
	��ֵ��Ŀ��patch�����50%���ģ�����Ϊ�京��ǰ��Ŀ��
	*/
	 else
	{
		for (int i = 0; i < grid.size(); i++){//FIXME: BottleNeck ƿ��  
			if (getVar(grid[i], iisum, iisqsum) >= var) //����ÿһ��ɨ�贰�ڵķ���
			{
				a++;
				//����������ģ��������Ϸ��������ģ�� 
				patch = img(grid[i]);
				classifier.getFeatures(patch, grid[i].sidx, ferns);//�õ���patch������13λ�Ķ�
				//���ƴ��룩  
				conf = classifier.measure_forest(ferns); //���������ֵ��Ӧ�ĺ�������ۼ�ֵ 
				tmp.conf[i] = conf; //Detector data�ж���TempStruct tmp; 
				tmp.patt[i] = ferns;

				//������Ϸ������ĺ�����ʵ�ƽ��ֵ������ֵfern_th����ѵ���õ�)������Ϊ����ǰ��Ŀ��
				if (conf > numtrees*fern_th)
				{
					dt.bb.push_back(i); //��ͨ�������������ģ���ɨ�贰�ڼ�¼��detect structure��
				}
			}
			else
				tmp.conf[i] = 0.0;
		}
		int detections = dt.bb.size();
		printf("%d Bounding boxes passed the variance filter\n", a);
		printf("%d Initial detection from Fern Classifier\n", detections);
		//���ͨ�������������ģ���ɨ�贰��������100������ֻȡ������ʴ��ǰ100�� 
		if (detections > 100){ //CComparator(tmp.conf)ָ���ȽϷ�ʽ������  
			nth_element(dt.bb.begin(), dt.bb.begin() + 100, dt.bb.end(), CComparator(tmp.conf));
			dt.bb.resize(100);
			detections = 100;
		}
		//��ʾ��⵽��bounding box
		  for (int i=0;i<detections;i++){
		        drawBox(img,grid[dt.bb[i]]);
		    }
		  imshow("ԭʼ detections",img);


		if (detections == 0){
			detected = false;
			return;
		}
		printf("Fern detector made %d detections ", detections);
		t = (double)getTickCount() - t; //����ʹ��getTickCount()��Ȼ���ٳ���getTickFrequency()��
		//���������������sΪ��λ��ʱ��
		printf("in %gms\n", t * 1000 / getTickFrequency());
		//  Initialize detection structure
		dt.patt = vector<vector<int> >(detections, vector<int>(10, 0));        //  Corresponding codes of the Ensemble Classifier
		dt.conf1 = vector<float>(detections);                                //  Relative Similarity (for final nearest neighbour classifier)
		dt.conf2 = vector<float>(detections);                                 //  Conservative Similarity (for integration with tracker)
		dt.isin = vector<vector<int> >(detections, vector<int>(3, -1));        //  Detected (isin=1) or rejected (isin=0) by nearest neighbour classifier
		dt.patch = vector<Mat>(detections, Mat(patch_size, patch_size, CV_32F));//  Corresponding patches
		int idx;
		Scalar mean, stdev;
		float nn_th = classifier.getNNTh();


		//����������ģ����������ڷ��������ģ��
		for (int i = 0; i < detections; i++)

		{                                         //  for every remaining detection
			idx = dt.bb[i];                                                       //  Get the detected bounding box index
			patch = frame(grid[idx]);
			getPattern(patch, dt.patch[i], mean, stdev);                //  Get pattern within bounding box
			//����ͼ��Ƭpattern������ģ��M��������ƶȺͱ������ƶ�
			classifier.NNConf(dt.patch[i], dt.isin[i], dt.conf1[i], dt.conf2[i]);  //  Evaluate nearest neighbour classifier
			dt.patt[i] = tmp.patt[idx];
			//printf("Testing feature %d, conf:%f isin:(%d|%d|%d)\n",i,dt.conf1[i],dt.isin[i][0],dt.isin[i][1],dt.isin[i][2]);
			//������ƶȴ�����ֵ������Ϊ����ǰ��Ŀ��
			if (dt.conf1[i] > nn_th)
			{                                               //  idx = dt.conf1 > tld.model.thr_nn; % get all indexes that made it through the nearest neighbour
				dbb.push_back(grid[idx]);                                         //  BB    = dt.bb(:,idx); % bounding boxes
				dconf.push_back(dt.conf2[i]);                                     //  Conf  = dt.conf2(:,idx); % conservative confidences
			}
		}
		
		//  end
		if (dbb.size() > 0)
		{ //��ӡ��⵽�Ŀ��ܴ���Ŀ���ɨ�贰����������ͨ����������������ģ�
			printf("Found %d NN matches\n", (int)dbb.size());
			detected = true;
		}
		else
		{
			printf("No NN matches found.\n");
			detected = false;
		}
	}
}



//�����Ѿ���python�ű�../datasets/evaluate_vis.py������㷨�������ܣ������README  
void TLD::evaluate(){
}

void TLD::learn(const Mat& img){
	printf("[Learning] ");
	///Check consistency //���һ����
	BoundingBox bb;
	bb.x = max(lastbox.x, 0);
	bb.y = max(lastbox.y, 0);
	bb.width = min(min(img.cols - lastbox.x, lastbox.width), min(lastbox.width, lastbox.br().x));
	bb.height = min(min(img.rows - lastbox.y, lastbox.height), min(lastbox.height, lastbox.br().y));
	Scalar mean, stdev;
	Mat pattern;
	//��һ��img(bb)��Ӧ��patch��size��������patch_size = 15*15��������pattern  
	getPattern(img(bb), pattern, mean, stdev);
	vector<int> isin;
	float dummy, conf;
	//��������ͼ��Ƭ����������Ŀ��box��������ģ��֮���������ƶ�conf  
 	classifier.NNConf(pattern, isin, conf, dummy);
	if (conf < 0.5) { //������ƶ�̫С�ˣ��Ͳ�ѵ�� 
		printf("Fast change..not training\n");
		lastvalid = false;
		return;
	}
	if (pow(stdev.val[0], 2) < var){  //�������̫С�ˣ�Ҳ��ѵ�� 
		printf("Low variance..not training\n");
		lastvalid = false;
		return;
	}
	if (isin[2] == 1){  //�������ʶ��Ϊ��������Ҳ��ѵ��  
		printf("Patch in negative data..not traing");
		lastvalid = false;
		return;
	}
	/// Data generation ��������
	for (int i = 0; i < grid.size(); i++){  //�������е�ɨ�贰����Ŀ��box���ص���  
		grid[i].overlap = bbOverlap(lastbox, grid[i]);
	}

	//���Ϸ�����
	vector<pair<vector<int>, int> > fern_examples;
	good_boxes.clear();
	bad_boxes.clear();


	getOverlappingBoxes(lastbox, num_closest_update); //�˺������ݴ����lastbox��


	//����֡ͼ���е�ȫ��������Ѱ����
	//��lastbox������С���������ƣ�
	//�ص�����󣩵�num_closest_update
	//�����ڣ�Ȼ�����Щ���� ����good_boxes����
	//(ֻ�ǰ������������������)
	//ͬʱ���ص���С��0.2�Ĺ���bad_boxs����
	if (good_boxes.size() > 0)
		generatePositiveData(img, num_warps_update); //�÷���ģ�Ͳ����������������ڵ�һ֡�ķ�������ֻ����10*10����
	else{
		lastvalid = false;
		printf("No good boxes..Not training");
		return;
	}
	fern_examples.reserve(pX.size() + bad_boxes.size());
	fern_examples.assign(pX.begin(), pX.end());
	int idx;
	for (int i = 0; i < bad_boxes.size(); i++){
		idx = bad_boxes[i];
		if (tmp.conf[idx] >= 1){ //���븺���������ƶȴ���1�������ƶȲ��ǳ���0��1֮����
			fern_examples.push_back(make_pair(tmp.patt[idx], 0));
		}
	}

	//����ڷ�����
	vector<Mat> nn_examples;
	nn_examples.reserve(dt.bb.size() + 1);
	nn_examples.push_back(pEx);
	for (int i = 0; i < dt.bb.size(); i++){
		idx = dt.bb[i];
		if (bbOverlap(lastbox, grid[idx]) < bad_overlap)
			nn_examples.push_back(dt.patch[i]);
	}
	/// Classifiers update ������ѵ��
	classifier.trainF(fern_examples, 2);
	classifier.trainNN(nn_examples);
	classifier.show(); //���������⣨����ģ�ͣ�������������������ʾ�ڴ�����
}


// ����ɨ�贰��
/*
���������ɨ�贰�ڵĲ��ԣ�ɨ�贰�ڲ���Ϊ��ߵ�10%���߶�����ϵ��Ϊ1.2���˺�������ȫ����ɨ�贰��grid��
������ÿһ��ɨ�贰���������Ŀ��box���ص��ȣ��ص��ȶ���Ϊ����box�Ľ��������ǵĲ����ıȡ�
*/
//��󽫽����Ĵ������grid�����ڣ����в������ݣ�grid��TLD���һ����Ա������
void TLD::buildGrid(const cv::Mat& img, //��ʾ֡
	const cv::Rect& box //�˹��趨�ĳ�ʼ�߽����
	)
{
	const float SHIFT = 0.1; //ɨ�贰�ڲ���Ϊ��ߵ�10%
	const float SCALES[] = {  //�߶�����ϵ��Ϊ1.2 ��0.16151*1.2=0.19381������21�ֳ߶ȱ任
		0.16151, 0.19381, 0.23257, 0.27908, 0.33490, 0.40188, 0.48225,
		0.57870, 0.69444, 0.83333, 1, 1.20000, 1.44000, 1.72800,
		2.07360, 2.48832, 2.98598, 3.58318, 4.29982, 5.15978, 6.19174 };
	int width, height, min_bb_side;
	//Rect bbox;
	BoundingBox bbox;
	Size scale;
	int sc = 0;
	for (int s = 0; s<21; s++)
	{
		width = round(box.width*SCALES[s]);//round��������
		height = round(box.height*SCALES[s]);
		min_bb_side = min(height, width);//bounding box��̵ı� 
		/*
		����ͼ��Ƭ��min_winΪ15*15���أ�����bounding box �в����õ��ģ�����box
		�����min_winҪ�����⣬�����ͼ��϶���bounding box Ҫ����
		*/
		if (min_bb_side < min_win || width > img.cols || height > img.rows)
			continue;
		scale.width = width;
		scale.height = height;
		scales.push_back(scale); //�Ѹó߶ȵĴ��ڴ���scales������������ɨ��ʱ���㣬�ӿ����ٶ�
		for (int y = 1; y < img.rows - height; y += round(SHIFT*min_bb_side))
		{ //�������ƶ�����
			for (int x = 1; x < img.cols - width; x += round(SHIFT*min_bb_side))
			{
				bbox.x = x;
				bbox.y = y;
				bbox.width = width;
				bbox.height = height;
				/*
				�жϴ����bounding box��Ŀ��߽���봫��ͼ���еĴ�ʱ���ڵ��ص��ȣ�
				�Դ���ȷ����ͼ�񴰿��Ƿ���Ŀ�ꡣ
				*/
				bbox.overlap = bbOverlap(bbox, BoundingBox(box));
				bbox.sidx = sc; //���ڵڼ����߶�
				grid.push_back(bbox); //grid ��TLD�ж��壺std::vector<BoundingBox> grid
				//�ѱ�λ�úͳ߶ȵ�ɨ�贰�ڴ���grid����
			}
		}
		sc++;
	}
}

/*
	�˺�����������bounding box���ص���
	�ص��ȵĶ���Ϊ������box�Ľ��������ǵĲ����ı�
	*/
float TLD::bbOverlap(const BoundingBox& box1, const BoundingBox& box2){
	//���ж����꣬�������Ƕ�û���ص��ĵط�����ֱ�ӷ���0
	if (box1.x > box2.x + box2.width) { return 0.0; }
	if (box1.y > box2.y + box2.height) { return 0.0; }
	if (box1.x + box1.width < box2.x) { return 0.0; }
	if (box1.y + box1.height < box2.y) { return 0.0; }

	float colInt = min(box1.x + box1.width, box2.x + box2.width) - max(box1.x, box2.x);
	float rowInt = min(box1.y + box1.height, box2.y + box2.height) - max(box1.y, box2.y);

	float intersection = colInt * rowInt; //�ص����ֵ����
	float area1 = box1.width*box1.height;
	float area2 = box2.width*box2.height;
	return intersection / (area1 + area2 - intersection);
}

/*
  ����ĺ������ݴ����box��Ŀ��߽�򣩣�����֡ͼ���е�ȫ��ɨ�贰���У���buildGrid���ɣ�Ѱ�����box������С
  ���������ƣ��ص�����󣩵�num_closest_init(10)�����ڣ�Ȼ�����Щ���ڹ�Ϊgood_boxes������ͬʱ���ص���
  С��0.2�Ĺ��뵽bad_boxes�������൱�ڶ�ȫ����ɨ�贰�ڽ���ɸѡ����ͨ��BBhull�����õ���Щɨ�贰�ڵ����߽硣
  */
void TLD::getOverlappingBoxes(const cv::Rect& box1, int num_closest)
{
	float max_overlap = 0;
//	if (firsttime||!tracked )//�ж��Ƿ��һ�ν���˺����������һ�Σ�ʹ��ȫ��ɨ�贰�ڲ���good_boxes �� Hull��
//	{
		for (int i = 0; i<grid.size(); i++)
		{
			if (grid[i].overlap > max_overlap)
			{ //�ҳ��ص�������box
				max_overlap = grid[i].overlap;
				best_box = grid[i];
			}
			if (grid[i].overlap > 0.6)
			{ //�ص��ȴ���0.6�ģ�����good_boxes
				good_boxes.push_back(i);
			}
			else if (grid[i].overlap < bad_overlap){ //�ص���С��0.2�ģ�����bad_boxes
				bad_boxes.push_back(i);
			}
		}
		//Get the best num_closest (10) boxes and puts them in good_boxes
		if (good_boxes.size()>num_closest)
		{
			/*
			STL��nth_element()�����ҳ�һ��������������n������Ϊ��num_closest���ĸ�����
			����������к���good_boxes[num_closest]ǰ��num_closest������������Ҳ����
			�ҵ���õ�Num_closest��box��
			*/
			std::nth_element(good_boxes.begin(), good_boxes.begin() + num_closest, good_boxes.end(), OComparator(grid));
			good_boxes.resize(num_closest);  //����ѹ��good_boxesΪnum_closest��С
		}
		getBBHull();  //��ȡgood_boxes �� Hull�ǣ�Ҳ���Ǵ��ڵı߿�
		firsttime = false;
//	}
	//�ǵ�һ��ѭ���������´��룬����koverlap�����ټ�����
/*	else if (tracked)
	{

		
		for (int i = 0; i<koverlapidx.size(); i++)
		{
			if (grid[koverlapidx[i]].overlap > max_overlap)
			{ //�ҳ��ص�������box
				max_overlap = grid[koverlapidx[i]].overlap;
				best_box = grid[koverlapidx[i]];
			}
			if (grid[koverlapidx[i]].overlap > 0.47)
			{ //�ص��ȴ���0.6�ģ�����good_boxes
				good_boxes.push_back(koverlapidx[i]);
			}
			else if (grid[koverlapidx[i]].overlap < bad_overlap){ //�ص���С��0.2�ģ�����bad_boxes
				bad_boxes.push_back(koverlapidx[i]);
			}
		}
		//Get the best num_closest (10) boxes and puts them in good_boxes
		if (good_boxes.size()>num_closest)
		{
			/*
			STL��nth_element()�����ҳ�һ��������������n������Ϊ��num_closest���ĸ�����
			����������к���good_boxes[num_closest]ǰ��num_closest������������Ҳ����
			�ҵ���õ�Num_closest��box��

			std::nth_element(good_boxes.begin(), good_boxes.begin() + num_closest, good_boxes.end(), OComparator(grid));
			good_boxes.resize(num_closest);  //����ѹ��good_boxesΪnum_closest��С
		}
		getBBHull();  //��ȡgood_boxes �� Hull�ǣ�Ҳ���Ǵ��ڵı߿�
	}*/
}

//�˺�����ȡgood_boxes �� Hull�ǣ�Ҳ���Ǵ��ڣ�ͼ�񣩵ı߿� bounding box  
void TLD::getBBHull(){
	int x1 = INT_MAX, x2 = 0;
	int y1 = INT_MAX, y2 = 0;
	int idx;
	for (int i = 0; i < good_boxes.size(); i++){
		idx = good_boxes[i];
		x1 = min(grid[idx].x, x1);
		y1 = min(grid[idx].y, y1);
		x2 = max(grid[idx].x + grid[idx].width, x2);
		y2 = max(grid[idx].y + grid[idx].height, y2);
	}
	bbhull.x = x1;
	bbhull.y = y1;
	bbhull.width = x2 - x1;
	bbhull.height = y2 - y1;
}

//�������box���ص���С��0.5������false�����򷵻�true  
bool bbcomp(const BoundingBox& b1, const BoundingBox& b2){
	TLD t;
	if (t.bbOverlap(b1, b2) < 0.5)
		return false;
	else
		return true;
}

int TLD::clusterBB(const vector<BoundingBox>& dbb, vector<int>& indexes){
	//FIXME: Conditional jump or move depends on uninitialised value(s)
	const int c = dbb.size();
	//1. Build proximity matrix
	Mat D(c, c, CV_32F);
	float d;
	for (int i = 0; i < c; i++){
		for (int j = i + 1; j < c; j++){
			d = 1 - bbOverlap(dbb[i], dbb[j]);
			D.at<float>(i, j) = d;
			D.at<float>(j, i) = d;
		}
	}
	//2. Initialize disjoint clustering
	// float L[c-1]; //Level
	float *L = new float[c - 1];
	// int nodes[c-1][2];
	int **nodes = new int *[c - 1];
	for (int i = 0; i < 2; i++)
		nodes[i] = new int[c - 1];

	// int belongs[c];
	int *belongs = new int[c];
	int m = c;
	for (int i = 0; i < c; i++){
		belongs[i] = i;
	}
	for (int it = 0; it < c - 1; it++){
		//3. Find nearest neighbor
		float min_d = 1;
		int node_a, node_b;
		for (int i = 0; i < D.rows; i++){
			for (int j = i + 1; j < D.cols; j++){
				if (D.at<float>(i, j)<min_d && belongs[i] != belongs[j]){
					min_d = D.at<float>(i, j);
					node_a = i;
					node_b = j;
				}
			}
		}
		if (min_d>0.5){
			int max_idx = 0;
			bool visited;
			for (int j = 0; j < c; j++){
				visited = false;
				for (int i = 0; i < 2 * c - 1; i++){
					if (belongs[j] == i){
						indexes[j] = max_idx;
						visited = true;
					}
				}
				if (visited)
					max_idx++;
			}
			return max_idx;
		}

		//4. Merge clusters and assign level
		L[m] = min_d;
		nodes[it][0] = belongs[node_a];
		nodes[it][1] = belongs[node_b];
		for (int k = 0; k < c; k++){
			if (belongs[k] == belongs[node_a] || belongs[k] == belongs[node_b])
				belongs[k] = m;
		}
		m++;
	}
	// �ͷ��ڴ�
	delete[] L;
	L = NULL;
	for (int i = 0; i < 2; ++i)
	{
		delete[] nodes[i];
		nodes[i] = NULL;
	}
	delete[]nodes;
	nodes = NULL;
	delete[] belongs;
	belongs = NULL;

	return 1;

}

/*
	�Լ������⵽��Ŀ��bounding box���о���
	���ࣨCluster��������������ģʽ��Pattern����ɵģ�
	ͨ����ģʽ��һ��������Measurement���������������Ƕ�ά�ռ��е�һ���㡣
	���������������Ϊ��������һ�������е�ģʽ֮��Ȳ���ͬһ�����е�ģʽ
	֮����и����������
	*/
void TLD::clusterConf(const vector<BoundingBox>& dbb, const vector<float>& dconf, vector<BoundingBox>& cbb, vector<float>& cconf){
	int numbb = dbb.size();
	vector<int> T;
	float space_thr = 0.5;
	int c = 1; //��¼ ����������  
	switch (numbb){ //��⵽�ĺ���Ŀ���bounding box���� 
	case 1:
		cbb = vector<BoundingBox>(1, dbb[0]); //���ֻ��⵽һ������ô������Ǽ������⵽��Ŀ��
		cconf = vector<float>(1, dconf[0]);
		return;
		break;
	case 2:
		T = vector<int>(2, 0);
		if (1 - bbOverlap(dbb[0], dbb[1]) > space_thr){  //�˺�����������bounding box ���ص���
			T[1] = 1;                               //���ֻ��⵽����box�������ǵ��ص���С��0.5
			c = 2;                                  //�ص���С��0.5��box�����ڲ�ͬ����
		}
		break;
	default: //��⵽��box��Ŀ����2������ɸѡ���ص��ȴ���0.5�� 
		T = vector<int>(numbb, 0);
		/*
		stable_partition()��������Ԫ�أ�ʹ������ָ��������Ԫ�����ڲ�����������Ԫ��ǰ�档
		��ά��������Ԫ�ص�˳���ϵ��
		STL partition���ǰ�һ�������е�Ԫ�ذ���ĳ�������ֳ����ࡣ���صڶ����Ӽ������
		bbcomp()�����ж�����box���ص���С��0.5������false�����򷵻�true ���ֽ�����ص��ȣ�0.5��
		partition() ��dbb����Ϊ�����Ӽ�������������box���ص���С��0.5��Ԫ���ƶ������е�ǰ�棬
		Ϊһ���Ӽ����ص��ȴ���0.5�ģ��������к��棬Ϊ�ڶ����Ӽ����������Ӽ��Ĵ�С��֪��������
		�ڶ����Ӽ������
		*/
		c = partition(dbb, T, (*bbcomp));//�ص���С��0.5��box�����ڲ�ͬ���࣬����c�ǲ�ͬ��������  
		//c = clusterBB(dbb,T);
		break;
	}
	cconf = vector<float>(c);
	cbb = vector<BoundingBox>(c);
	printf("Cluster indexes: ");
	BoundingBox bx;
	for (int i = 0; i < c; i++){ //������
		float cnf = 0;
		int N = 0, mx = 0, my = 0, mw = 0, mh = 0;
		for (int j = 0; j < T.size(); j++){ //��⵽��bounding box����
			if (T[j] == i)
			{  //������Ϊͬһ������box������ʹ�С�����ۼ� 
				printf("%d ", i);
				cnf = cnf + dconf[j];
				mx = mx + dbb[j].x;
				my = my + dbb[j].y;
				mw = mw + dbb[j].width;
				mh = mh + dbb[j].height;
				N++;
			}
		}
		if (N > 0){ //Ȼ��������box������ʹ�С��ƽ��ֵ����ƽ��ֵ��Ϊ�����box�Ĵ���  
			cconf[i] = cnf / N;
			bx.x = cvRound(mx / N);
			bx.y = cvRound(my / N);
			bx.width = cvRound(mw / N);
			bx.height = cvRound(mh / N);
			cbb[i] = bx;  //���ص��Ǿ��࣬ÿһ���඼��һ�������bounding box
		}
	}
	printf("\n");
}

