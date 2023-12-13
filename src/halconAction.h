#ifndef _HALCONACTION__H_
#define _HALCONACTION__H_

#include "spdlog/sinks/basic_file_sink.h"
#include "spdlog/sinks/stdout_sinks.h"
#include "spdlog/async.h"
#include "spdlog/sinks/rotating_file_sink.h"
#include "thread"

#include <opencv2/opencv.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaarithm.hpp>

#include <torch/torch.h>
#include <fstream>

#include "stdio.h"

#include "HDevThread.h"
#include "HDevEngineCpp.h"
#include "CommonContent.h"


using namespace cv;
using namespace std;
using namespace HalconCpp;
using namespace HDevEngineCpp;


struct hconAction_middle_result {

	//HalconCpp::HTuple hv_Row11;
	//HalconCpp::HTuple hv_Column11;
	//HalconCpp::HTuple hv_Row2;
	//HalconCpp::HTuple hv_Column2;

	//HalconCpp::HTuple AreaSumModel;  //模板的连通面积
	//HalconCpp::HTuple AreaSumModelDila;

	//HalconCpp::HObject ho_RegionDifference;

	HalconCpp::HObject ho_RegionIntersection;
	HalconCpp::HTuple hv_ModelID;

	//HalconCpp::HTuple hv_RowLoc;
	//HalconCpp::HTuple hv_ColLoc;

	HalconCpp::HTuple hv_RefRow;
	HalconCpp::HTuple hv_RefCol;

	HalconCpp::HObject ho_TemplateImageM;
	HalconCpp::HObject ho_RegionDilationM;
	HalconCpp::HObject ho_RegionM;

	HalconCpp::HObject ho_ModelRegion;
};

//计算过程参数
struct hconAction_calculate {
	int delta_x = 0;
	int delta_y = 0;

	HalconCpp::HTuple hv_XM;
	HalconCpp::HTuple hv_YM;

	int ThresholdVal;   //配置文件中的参数,所有模板共用的参数
};

struct hconAction_moban {

	hconAction_middle_result mid_result; //中间结果

	int now_num = 0;

	int x1 = 0;      //模板坐标
	int y1 = 0;
	int x2 = 0;
	int y2 = 0;

	int BlackMin = 0;
	int WhiteMin = 0;
	int BlackMax = 45;
	int WhiteMax = 0;

	float RatioDila = 0;  //结果要 /10
	float DiffThr = 0;		//结果要 /10
	float DiffThrD = 33;   //结果要 /10

	int MinArea = 0;
	int MaxArea = 0;

	int XOffSet = 0;
	int YOffSet = 0;

	int Mult = 0;   //放大倍数
	int Add = 0;    //放大后的偏移

	//刀丝检测参数
	float cannyAlpha = 0; //结果要 /10
	float cannyLow = 0; //结果要 /10
	float cannyHigh = 0; //结果要 /10
	float adjacentDisAbs = 0; //结果要 /10
	float adjacentDisRel = 0; //结果要 /10
	int contourMin = 0;
	int contourMax = 0;
};

class hconAction
{
public:
	explicit hconAction();
	~hconAction();

public:
	Mat HImageToMat(const HalconCpp::HImage& hImg);   //halcon格式图片转换为opencv图片
	Mat HObjectToMat(const HalconCpp::HObject& hObj);
	HalconCpp::HObject MatToHObject(const Mat& image);
	HalconCpp::HImage MatToHImage(const Mat& image);
	void SaveHobjectImg(string path, HObject& Hobject_in);

	void HALCON_Refresh_Model_all(string filePath_moban); //更新模板文件缓存
	void ImageThreadStart(unsigned char cameraid); //开启模板比对的线程
	void ImageThreadJoin(vector<Detect_Result>& Result, int& is_detect);
	//void ImageThreadRun(int threadnum, hconAction_calculate& calculate, hconAction_moban& Model, HalconCpp::HObject& ho_TransImage, unsigned int now_img_num);

	void GetDataOfmoban(int num, int& x1, int& y1, int& x2, int& y2);  //获取模板坐标数据
	int GetNumOfmoban();
	void SetNowImgNum(int num_now);   //设置当前正在处理第几张图片

	void GetQrInfo(unsigned char cameraid, string& qr_str, float qr_danbian_x, float qr_danbian_y, int& result);//获取当前的二维码信息,包括二维码数据和二维码单边
	void GetDanBianInfo(unsigned char cameraid, float danbian_config, int& result);//获取单边检测结果


	int GetIni(string file_path, string apname, string partname);     //获取参数
	int SetIni(string file_path, string apname, string partname, int set_num);   //设置参数

	int ImageChange(HalconCpp::HObject& ho_ImageInput, HalconCpp::HObject& ho_TransImage1);//变换


	void ImgPreDealThread(cv::Mat& oriMatImage);
	int ImgPreDealJoin();
	void ImgPreDeal(cv::Mat& oriMatImage);

	std::thread *ImageChange_thread;
	HalconCpp::HObject ho_ImageChangeResult;
	int ImageChange_result = 0;

	unsigned int now_dealImg_num = 0;

	vector<std::thread> Halcon_thread;
	vector<hconAction_moban> Halcon_moban_all;
	hconAction_calculate calculate;   //变换前后的坐标差值

	

	//HalconCpp::HTuple hv_ModelX, hv_ModelY, hv_ModelNumber;

private:


};



#endif
