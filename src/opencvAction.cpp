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

//���ش�С��ֵ�ж�
bool opencvAction::cv_zhadianCalculate(cv::Mat &img, int yuzhi, int framenum)
{
    std::clock_t t1, t2;
    t1 = clock();
    // ת��Ϊ�Ҷ�ͼ
    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    // ��ֵ��
    cv::Mat binary;
    cv::threshold(gray, binary, 0, 255, cv::THRESH_BINARY + cv::THRESH_OTSU);
    // �������
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(binary, contours, hierarchy, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE);

    double area_sum = 0;
    for (size_t i = 0; i < contours.size(); i++) {
        double area = cv::contourArea(contours[i]);  // �������������
        // �����ͨ��������areaС��ͼƬ�Ŀ�*�ߣ�����������ۼ�
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

//������
cv::Mat opencvAction::CV_drawRectangle(cv::Mat& img, int x1, int y1, int x2, int y2, int class_1, int danbian_x,int danbian_y1, int danbian_y2, int danbian_width, int danbian_height)
{
    string class_name = to_string(class_1);
    int delta_x = 0;
    delta_x = abs(x1 - x2) / 2;
    int delta_y = 0 ;
    delta_y = abs(y1 - y2) / 2;
    if (class_1 > 50 || class_1 == 23 || class_1 == 5) {   //���߻��߶�ά��ȱ�� ��ī �����Ŵ���
        delta_x = 0;
        delta_y = 0;
    }
    if (x1 != 0 && x2 != 0 && y1 != 0 && y2 != 0) {
        //ȱ������ x1<x2   y1<y2
        cv::rectangle(img, cv::Point(x1 - delta_x, y1 - delta_y), cv::Point(x2 + delta_x, y2 + delta_y), Scalar(0, 0, 255), 2, 8, 0);
        putText(img, class_name, Point(x1 - delta_x - 30, y1 - delta_y - 5), FONT_HERSHEY_PLAIN, 2, Scalar(0, 0, 255), 2);
    }
    else if (class_1 == 16 && x1 == 0 && x2 == 0) {   //������Ҫ�����������
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

//ͼ������
cv::Mat opencvAction::OpencvResize(cv::Mat matSend, float scaleW, float scaleH)
{
    //ͼ������
    //float scaleW = 0.65; // ������ͼ��Ĵ�С�������С��65%
    //float scaleH = 0.65;  //������ͼ��Ĵ�С���߶���С��65%
    int width = int(matSend.cols * scaleW);
    //������Ҫ���������С��Ŀ�ȣ�src.colsΪԭͼ��Ŀ�ȣ�����80%��õ���Ҫ�Ĵ�С����ǿ��ת����int��
    int height = int(matSend.rows * scaleH);
    //������Ҫ���������С��ĸ߶ȣ�src.colsΪԭͼ��ĸ߶ȣ�����80%��õ���Ҫ�Ĵ�С����ǿ��ת����int��
    cv::Mat matSend_1;
    resize(matSend, matSend_1, cv::Size(width, height));//����ͼ��
    return matSend_1;
}

// ��������������ټ�һ��format��ʽ������push��vectorǰ�ж����ļ�����׺��������ָ����ʽ
void opencvAction::GetAllFiles(std::string path, std::vector<std::string>& files) {
    // �����洢�ļ���Ϣ�Ľṹ�壬��ͷ�ļ� <io.h>
    struct _finddata_t fileinfo;
    intptr_t hFile = 0;
    std::string p;  // ������p(path)��ʼ��
    // ��һ�β���
    if ((hFile = _findfirst(p.assign(path).append("\\*").c_str(), &fileinfo)) != -1) {
        do {
            // ����ҵ������ļ���
            if ((fileinfo.attrib & _A_SUBDIR)) {
                continue;
                // ��������ļ��У���������continue
                if (std::strcmp(fileinfo.name, ".") != 0 && std::strcmp(fileinfo.name, "..") != 0) {
                    // �������
                    files.push_back(p.assign(path).append("\\").append(fileinfo.name));
                    GetAllFiles(p.assign(path).append("\\").append(fileinfo.name), files);
                }
            }
            // ����ҵ��Ĳ����ļ���
            else {
                // ��������ļ���
                files.push_back(p.assign(fileinfo.name));
                // Ҳ�����Ǳ������·��
                // files.push_back(p.assign(path).append("\\").append(fileinfo.name));  
            }
        } while (_findnext(hFile, &fileinfo) == 0);
        // ��������
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

    if (x1 + width > inpic.cols) x1 = inpic.cols - width;   //��ֹԽ��
    if (y1 + height > inpic.rows) y1 = inpic.rows - height;


    outpic = inpic(cv::Rect(x1, y1, width, height));
}
