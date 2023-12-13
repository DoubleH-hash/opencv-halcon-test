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

#include "Halcon.h"
#include "HalconCpp.h"
#include "HDevThread.h"
#include "HDevEngineCpp.h"

#include "halconAction.h"
#include "opencvAction.h"

using namespace cv;
using namespace std;
using namespace HDevEngineCpp;

#pragma comment(lib,"halconcpp.lib")

using namespace HalconCpp;

opencvAction opencv_Action;

hconAction halcon_Action;

void init_log()
{
    try
    {
		spdlog::init_thread_pool(10000, 1);
        char log_file_name[256];
        DWORD process_id = GetCurrentProcessId();
        sprintf(log_file_name, "logs/basic-2d-log-%ld.txt", process_id);
        //printf("file name : %s", log_file_name);
		auto logger = spdlog::rotating_logger_mt<spdlog::async_factory>("basic_logger",string(log_file_name), 1024 * 1024 * 50, 3);
        //auto logger = spdlog::basic_logger_mt("basic_logger", "logs/basic-log.txt");
        spdlog::set_default_logger(logger);
        // change log pattern
        spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e]> %v");
        SPDLOG_INFO("2d_hikvision node global log create successful");
        spdlog::flush_every(std::chrono::seconds(5));

        spdlog::flush_on(spdlog::level::info);

        std::cout << "Log init succeed!" << std::endl;
    }
    catch (const spdlog::spdlog_ex& ex)
    {
        std::cout << "Log init failed! " << ex.what() << std::endl;
    }
}

struct Result_prediction {
    string img_path;
    int usetime = 0;
    int result;
};

//样品图片
cv::Mat temp_img;
//结果分析
Result_prediction temp_result;
vector<Result_prediction> Result_analsys;
std::mutex result_mutex;

int product_class = 1103;   //黄鹤楼1916
int camera_now = 2;  //当前是第几个相机 1~5 
string HALCON_filePath_moban;//模板目录


string HALCON_filePath_templete;//样品目录


int main(int argc, char* argv[])
{
    system("chcp 65001");
    init_log();
    SPDLOG_INFO("Spdlog version {}.{}.{}", SPDLOG_VER_MAJOR,SPDLOG_VER_MINOR,SPDLOG_VER_PATCH);
    spdlog::set_level(spdlog::level::info);
    
    //模板位置
    HALCON_filePath_moban = "D:/HALCON/moban/camera" + to_string(camera_now) + "/" + to_string(product_class) + "/";

    halcon_Action.HALCON_Refresh_Model_all(HALCON_filePath_moban);

    //样品位置
    //string img_filePath = "D:/OpenCV_test/OpenCV/out/build/x64-Release/Testing/result-moban/";//样品目录
    //string img_filePath = "D:/OpenCV_test/OpenCV/out/build/x64-Release/Testing/zhengtu-bad-1115/B/";//样品目录
    //string img_filePath = "D:/OpenCV_test/OpenCV/out/build/x64-Release/Testing/camera2/";//样品目录
    //string img_filePath = "D:/OpenCV_test/OpenCV/out/build/x64-Release/Testing/zhengtu-bad-1129/";//样品目录
    //string img_filePath = "D:/OpenCV_test/OpenCV/out/build/x64-Release/Testing/0829haopian/";//样品目录
    string img_filePath = "D:/OpenCV_test/OpenCV/out/build/x64-Release/Testing/test--1212/";//样品目录
    //string img_filePath = "D:/OpenCV_test/OpenCV/out/build/x64-Release/Testing/jiduanceshi/";//样品目录
    //string img_filePath = "D:/OpenCV_test/OpenCV/out/build/x64-Release/Testing/camera2-1211/";//样品目录
    vector<string> files;
    //获取该路径下的所有文件
    opencv_Action.GetAllFiles(img_filePath, files);
    int imgfile_size = files.size();

    int start_testnum = 0;
    int start_timesum = 0;
start:

    int imgnum = 0;
    int usetime_sum = 0;

    while (imgnum < imgfile_size) {
        std::clock_t t1, t2;

        vector<Detect_Result> ALL_Result;

        string filePath = img_filePath + files[imgnum];

        halcon_Action.SetNowImgNum(imgnum);

        cv::Mat oriMatImage = cv::imread(filePath);

        t1 = clock();

        int result = 0;
        if (halcon_Action.GetNumOfmoban() != 0) {
            halcon_Action.ImgPreDealThread(oriMatImage);

            result = halcon_Action.ImgPreDealJoin();
        }

        if (result == 1) {

            //保存变换的结果
            //halcon_Action.SaveHobjectImg("D:/OpenCV_test/OpenCV/out/build/x64-Release/Testing/result/" + to_string(imgnum) + "_1.bmp", ImageInput_2);
            halcon_Action.ImageThreadStart(camera_now);
        }
        else if(result == -1){   //定位失败
            Detect_Result result_halcon;
            result_halcon.class_res = 4;   //窜墨
            result_halcon.x1 = 122;
            result_halcon.x2 = 2348;
            result_halcon.y1 = 135;
            result_halcon.y2 = 854;
            result_halcon.result_str = "{\"class\":" + to_string(result_halcon.class_res) + ",\"config\":99" +
                +",\"ix1\":" + to_string(result_halcon.x1) + ",\"iy1\":" + to_string(result_halcon.y1)
                + ",\"ix2\":" + to_string(result_halcon.x2) + ",\"iy2\":" + to_string(result_halcon.y2) + "}";
            ALL_Result.push_back(result_halcon);
        }


        int HALCON_is_detect = 0;
        halcon_Action.ImageThreadJoin(ALL_Result, HALCON_is_detect);

        //读取二维码信息
        string qr_code_res;
        int qr_result = 0;
        halcon_Action.GetQrInfo(camera_now, qr_code_res, 2.5, 2.5, qr_result);
        if (qr_code_res.size() < 5) {
            std::cout << "can't find qr" << std::endl;
        }else {
            std::cout << "qr:" << qr_code_res << ",qr_result:" << qr_result << std::endl;
        }

        int danbian_result = 0;
        halcon_Action.GetDanBianInfo(camera_now, 4, danbian_result);
        if(danbian_result == 1){
            std::cout << "detect danbian !" << std::endl;
        }
        t2 = clock();

        string ImageOut_path_1;
        if (HALCON_is_detect == 1) {
            std::cout << "file:" << files[imgnum] << ",ALL_Result size:" << to_string(ALL_Result.size()) << ",x1:" << to_string(ALL_Result[0].x1) << ",y1:" << to_string(ALL_Result[0].y1) <<",usetime:" << to_string(t2 - t1) << std::endl;
            std::cout << "" << std::endl;
            ImageOut_path_1 = "D:/OpenCV_test/OpenCV/out/build/x64-Release/Testing/result/NG/";
        }
        else {
            std::cout << "file:" << files[imgnum] << ",OK" << ",usetime:" << to_string(t2 - t1) << std::endl;
            std::cout << "" << std::endl;
            ImageOut_path_1 = "D:/OpenCV_test/OpenCV/out/build/x64-Release/Testing/result/OK/";
        }
        

        string ImageOut_path = ImageOut_path_1 + files[imgnum];
        cv::Mat cv_result_img;
        cv_result_img = imread(filePath);
        
        unsigned int result_size_AllDefect = ALL_Result.size();
        unsigned int now_deal_num = 0;  //当前处理的缺陷数
        cv::Mat output_img;

        if(result_size_AllDefect >= 2) opencv_Action.DealResult(ALL_Result, result_size_AllDefect);

        //if (result == 1) cv_result_img = halcon_Action.HObjectToMat(halcon_Action.ho_ImageChangeResult);
        //else cv_result_img = imread(filePath);
        int add_x = 20, add_y = 20;
        for (int i = 0; i < ALL_Result.size(); i++) {
            cv_result_img = opencv_Action.CV_drawRectangle(cv_result_img, ALL_Result[i].x1 - add_x, ALL_Result[i].y1 - add_y, ALL_Result[i].x2 + add_x, ALL_Result[i].y2 + add_y, ALL_Result[i].class_res, 0, 0, 0, 0, 0);
        }
        imwrite(ImageOut_path, cv_result_img);

        while(now_deal_num < result_size_AllDefect){
            opencv_Action.CutPic(ALL_Result[now_deal_num], cv_result_img, output_img);

            ImageOut_path = ImageOut_path_1 + files[imgnum]  + "__" + to_string(now_deal_num) +".bmp";
            imwrite(ImageOut_path, output_img);

            now_deal_num++;
        }




        usetime_sum += t2 - t1;

        imgnum++;
    }
    




    //将模板显示出来
    string ImageModel_path = HALCON_filePath_moban + "0.bmp";
    string ImageOut_path = HALCON_filePath_moban + "out.bmp";
    cv::Mat cv_moban_img;
    cv_moban_img = imread(ImageModel_path);
    int x1, y1, x2, y2;

    for (int i = 1; i <= halcon_Action.GetNumOfmoban(); i++) {

        halcon_Action.GetDataOfmoban(i, x1,y1,x2,y2);
        cv_moban_img = opencv_Action.CV_drawRectangle(cv_moban_img, x1, y1, x2, y2,i,0,0,0,0,0);
    }
    imwrite(ImageOut_path , cv_moban_img);
    

    std::cout << "finish all! img_num:" << to_string(imgnum) << ",all usetime:" << to_string(usetime_sum / imgnum) << std::endl;


    //start_testnum++;
    //start_timesum += usetime_sum / imgnum;
    //if (start_testnum < 20) {
    //    goto start;
    //}
    //else {
     //   std::cout << "all test finsh! start_testnum:" << to_string(start_testnum) << ",all usetime:" << to_string(start_timesum / start_testnum) << std::endl;
    //}

    //while (1);

    return 0;
}