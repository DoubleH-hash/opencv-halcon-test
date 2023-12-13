#include <iostream>
#include <cstring>
#include <vector>
#include <io.h>
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

#include <fstream>

#include "stdio.h"

#include "Yolo.h"
#include "opencvAction.h"
#include "CommonContent.h"

using namespace cv;

using namespace std;


opencvAction::opencvAction()
{
    SPDLOG_INFO("opencvAction Init!");
}

opencvAction::~opencvAction()
{
    SPDLOG_INFO("opencvAction close!");
}

vector<cv::cuda::GpuMat> opencvAction::SlittingImage(const cv::cuda::GpuMat& matBody, const int& iNum)
{
    vector<cv::cuda::GpuMat> vctImageMat;

    int iImageWidth = (matBody.cols) / iNum;

    for (int i = 0; i < iNum; i++)
    {
        int iMatPartEnd = iImageWidth * (i + 1);
        if (matBody.cols < iMatPartEnd)
        {
            iMatPartEnd = matBody.cols;
        }
        cv::cuda::GpuMat matPart = matBody.colRange(iImageWidth * i, iImageWidth * (i + 1)).rowRange(0, matBody.rows).clone();
        //SPDLOG_INFO("Get Gpumat::matPart: {:d} cols:{:d} rows:{:d}", i, matPart.cols, matPart.rows);
        vctImageMat.push_back(matPart);
    }

    return vctImageMat;
}

//像素大小阈值判断
bool opencvAction::cv_zhadianCalculate(cv::Mat &img, int yuzhi, int framenum)
{
    std::clock_t t1, t2;
    t1 = clock();
    // 转换为灰度图
    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    // 二值化
    cv::Mat binary;
    cv::threshold(gray, binary, 0, 255, cv::THRESH_BINARY + cv::THRESH_OTSU);
    // 轮廓检测
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(binary, contours, hierarchy, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE);

    double area_sum = 0;
    for (size_t i = 0; i < contours.size(); i++) {
        double area = cv::contourArea(contours[i]);  // 计算轮廓的面积
        // 如果连通区域的面积area小于图片的宽*高，则将面积进行累加
        if (area <= gray.size[0] * gray.size[1]) ;
        area_sum += area;
    }

    t2 = clock();
    SPDLOG_INFO("cv_zhadianCalculate area_sum:{:f},usetime:{:d}", area_sum, t2 - t1);

    if (area_sum > yuzhi) {
        //std::cout << "NG" << std::endl;
        //string filepath = "D:/OpenCV_test/OpenCV/out/build/x64-release/Testing/result/NG/" + to_string(framenum) + ".bmp";
        //imwrite(filepath, img);

        return 1;
    }
    else {
        //std::cout << "OK" << std::endl;
        //string filepath = "D:/OpenCV_test/OpenCV/out/build/x64-release/Testing/result/OK/" + to_string(framenum) + ".bmp";
        //imwrite(filepath, img);
        return 0;
    }
}

void opencvAction::GetNowPredictionImg(int part, cv::cuda::GpuMat& vctMat, int qr_Usingflag, float danbian_config_dis, cv::Mat* out_img)
{
    switch (part) {
    case 0:
        if (danbian_config_dis < 0.1) {
            vctMat.download(out_img[0]);
        }
        break;
    case 1:
        vctMat.download(out_img[1]);
        break;
    case 2:
        if (qr_Usingflag == 0) {
            vctMat.download(out_img[2]);
        }
        break;
    default:break;
    }
}

//画矩形
cv::Mat opencvAction::CV_drawRectangle(cv::Mat& img, int x1, int y1, int x2, int y2, int class_1, int danbian_x,int danbian_y1, int danbian_y2, int danbian_width, int danbian_height)
{
    string class_name = to_string(class_1);
    int delta_x = 0;
    delta_x = abs(x1 - x2) / 2;
    int delta_y = 0 ;
    delta_y = abs(y1 - y2) / 2;
    if (class_1 > 50 || class_1 == 23 || class_1 == 5) {   //单边或者二维码缺陷 窜墨 不做放大处理
        delta_x = 0;
        delta_y = 0;
    }
    if (x1 != 0 && x2 != 0 && y1 != 0 && y2 != 0) {
        //缺陷数据 x1<x2   y1<y2
        cv::rectangle(img, cv::Point(x1 - delta_x, y1 - delta_y), cv::Point(x2 + delta_x, y2 + delta_y), Scalar(0, 0, 255), 2, 8, 0);
        putText(img, class_name, Point(x1 - delta_x - 30, y1 - delta_y - 5), FONT_HERSHEY_PLAIN, 2, Scalar(0, 0, 255), 2);
    }
    else if (class_1 == 16 && x1 == 0 && x2 == 0) {   //单边需要画两个框出来
        cv::rectangle(img, cv::Point(danbian_x, danbian_y1), cv::Point(danbian_x + danbian_width, danbian_y1 + danbian_height), Scalar(0, 0, 255), 2, 8, 0);
        cv::rectangle(img, cv::Point(danbian_x, danbian_y2), cv::Point(danbian_x + danbian_width, danbian_y2 + danbian_height), Scalar(0, 0, 255), 2, 8, 0);
        putText(img, class_name, Point(danbian_x - 30, danbian_y1 - 5), FONT_HERSHEY_PLAIN, 2, Scalar(0, 0, 255), 2);
    }
    SPDLOG_INFO("CV_drawRectangle {:s} x1:{:d},y1:{:d},x2:{:d},y2:{:d}", class_name, x1, y1, x2, y2);
    return img;
}

cv::cuda::GpuMat opencvAction::letterbox(cv::cuda::GpuMat& src, int h, int w, std::vector<int>& vctMakeBorder)
{
    float in_w = src.cols; // width
    float in_h = src.rows; // height
    float tar_w = w;
    float tar_h = h;
    float r = std::min(float(tar_h) / in_h, float(tar_w) / in_w);
    float inside_w = round(in_w * r);
    float inside_h = round(in_h * r);
    float pad_w = tar_w - inside_w;
    float pad_h = tar_h - inside_h;

    cv::cuda::GpuMat resize_img;

    cv::cuda::resize(src, resize_img, cv::Size(inside_w, inside_h));

    pad_w = pad_w / 2;
    pad_h = pad_h / 2;

    int top = int(round(pad_h - 0.1));
    int bottom = int(round(pad_h + 0.1));
    int left = int(round(pad_w - 0.1));
    int right = int(round(pad_w + 0.1));
    vctMakeBorder = { top, bottom , left , right };
    cv::cuda::copyMakeBorder(resize_img, resize_img, top, bottom, left, right, 0, cv::Scalar(114, 114, 114));

    return resize_img;
}

//图像缩放
cv::Mat opencvAction::OpencvResize(cv::Mat matSend, float scaleW, float scaleH)
{
    //图像缩放
    //float scaleW = 0.65; // 定义新图像的大小，宽度缩小到65%
    //float scaleH = 0.65;  //定义新图像的大小，高度缩小到65%
    int width = int(matSend.cols * scaleW);
    //定义想要扩大或者缩小后的宽度，src.cols为原图像的宽度，乘以80%则得到想要的大小，并强制转换成int型
    int height = int(matSend.rows * scaleH);
    //定义想要扩大或者缩小后的高度，src.cols为原图像的高度，乘以80%则得到想要的大小，并强制转换成int型
    cv::Mat matSend_1;
    resize(matSend, matSend_1, cv::Size(width, height));//缩放图像
    return matSend_1;
}

// 可在这个函数中再加一个format格式参数，push到vector前判断下文件名后缀，仅保留指定格式
void opencvAction::GetAllFiles(std::string path, std::vector<std::string>& files) {
    // 用来存储文件信息的结构体，在头文件 <io.h>
    struct _finddata_t fileinfo;
    intptr_t hFile = 0;
    std::string p;  // 不在这p(path)初始化
    // 第一次查找
    if ((hFile = _findfirst(p.assign(path).append("\\*").c_str(), &fileinfo)) != -1) {
        do {
            // 如果找到的是文件夹
            if ((fileinfo.attrib & _A_SUBDIR)) {
                continue;
                // 不想进入文件夹，就在这里continue
                if (std::strcmp(fileinfo.name, ".") != 0 && std::strcmp(fileinfo.name, "..") != 0) {
                    // 进入查找
                    files.push_back(p.assign(path).append("\\").append(fileinfo.name));
                    GetAllFiles(p.assign(path).append("\\").append(fileinfo.name), files);
                }
            }
            // 如果找到的不是文件夹
            else {
                // 保存的是文件名
                files.push_back(p.assign(fileinfo.name));
                // 也可以是保存绝对路径
                // files.push_back(p.assign(path).append("\\").append(fileinfo.name));  
            }
        } while (_findnext(hFile, &fileinfo) == 0);
        // 结束查找
        _findclose(hFile);
    }
}

void opencvAction::DealResult(vector<Detect_Result> &Result,unsigned int &Result_size)
{

    for (int i = 0; i < Result_size - 1; i++) {
        if ((abs(Result[i].x1 - Result[i + 1].x1) < 200) && (abs(Result[i].y1 - Result[i + 1].y1) < 200)) {
            Result[i + 1].x1 = (Result[i].x1 + Result[i + 1].x1 )/2;
            Result[i + 1].y1 = (Result[i].y1 + Result[i + 1].y1) / 2;
            Result[i + 1].x2 = (Result[i].x2 + Result[i + 1].x2) / 2;
            Result[i + 1].y2 = (Result[i].y2 + Result[i + 1].y2) / 2;
            Result.erase(Result.begin() + i);
            Result_size --;
            i--;
        }
    }
}

void opencvAction::CutPic(Detect_Result &Result, Mat &inpic, Mat &outpic)
{
     int x1 = 0, y1 = 0, width = 0, height = 0;

    x1 = Result.x1 - 200;
    if (x1 < 0) x1 = 0;
    y1 = Result.y1 - 200;
    if (y1 < 0) y1 = 0;
    width = 400;
    height = 400;

    if (x1 + width > inpic.cols) x1 = inpic.cols - width;   //防止越界
    if (y1 + height > inpic.rows) y1 = inpic.rows - height;


    outpic = inpic(cv::Rect(x1, y1, width, height));
}
