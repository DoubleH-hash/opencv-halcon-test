#ifndef __OPENCVACTION_H_
#define __OPENCVACTION_H_

#include <iostream>
#include <cstring>
#include <vector>

#include <opencv2/opencv.hpp>
#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_TRACE
#include "spdlog/spdlog.h"
#include "spdlog/cfg/env.h"
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

#include "Yolo.h"

#include "CommonContent.h"

using namespace cv;

using namespace std;


class opencvAction
{
public:
	explicit opencvAction();
	~opencvAction();

public:
	vector<cv::cuda::GpuMat> SlittingImage(const cv::cuda::GpuMat& matBody, const int& iNum);
	bool cv_zhadianCalculate(cv::Mat& img, int yuzhi, int framenum);
	void GetNowPredictionImg(int part, cv::cuda::GpuMat& vctMat, int qr_Usingflag, float danbian_config_dis, cv::Mat* out_img);
	cv::Mat CV_drawRectangle(cv::Mat& img, int x1, int y1, int x2, int y2, int class_1, int danbian_x, int danbian_y1, int danbian_y2, int danbian_width, int danbian_height);
	cv::cuda::GpuMat letterbox(cv::cuda::GpuMat& src, int h, int w, std::vector<int>& vctMakeBorder);
	cv::Mat OpencvResize(cv::Mat matSend, float scaleW, float scaleH);
	//bool FilePathCheck(string strImagePath);   //检测文件目录是否存在 若不存在则创建一个
	void GetAllFiles(std::string path, std::vector<std::string>& files);   //获取文件下的所有文件名

	void DealResult(vector<Detect_Result>& Result,unsigned int &Result_size);
	void CutPic(Detect_Result& Result, Mat& inpic, Mat &outpic);  //分析结果，将结果数据进行切图

private:


};



#endif
