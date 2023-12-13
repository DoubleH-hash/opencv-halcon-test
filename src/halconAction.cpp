#include <iostream>
#include <cstring>
#include <vector>
#include "thread"

#include <opencv2/opencv.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaarithm.hpp>
#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_TRACE
#include "spdlog/spdlog.h"
#include "spdlog/cfg/env.h"
#include "spdlog/sinks/basic_file_sink.h"
#include "spdlog/sinks/stdout_sinks.h"
#include "spdlog/async.h"
#include "spdlog/sinks/rotating_file_sink.h"

#include <torch/torch.h>
#include <fstream>

#include "stdio.h"
#include <future>

#include "windows.h"

#include "Halcon.h"
#include "HalconCpp.h"

#include "halconAction.h"
#include "HDevThread.h"
#include "HDevEngineCpp.h"

#include "CommonContent.h"

using namespace cv;
using namespace std;
using namespace HalconCpp;
using namespace HDevEngineCpp;

std::mutex halcon_result_mutex;
vector<Detect_Result> Halcon_Result;

HTuple hv_DataCodeHandle;  //二维码句柄
string qr_code_str; //二维码字符串
HalconCpp::HTuple hv_qr_xx, hv_qr_yy;   //二维码单边的计算结果

HTuple  hv_Deviation;   //单边的计算结果  

HTuple hv_DeviceIdentifiers, hv_DeviceHandle;


void OpenGPU()
{
    QueryAvailableComputeDevices(&hv_DeviceIdentifiers);
    hv_DeviceHandle = 0;
    {
        HTuple end_val87 = (hv_DeviceIdentifiers.TupleLength()) - 1;
        HTuple step_val87 = 1;
        HTuple hv_i, hv_Nmae;
        for (hv_i = 0; hv_i.Continue(end_val87, step_val87); hv_i += step_val87) {
            GetComputeDeviceInfo(HTuple(hv_DeviceIdentifiers[hv_i]), "name", &hv_Nmae);
            if (0 != (int(hv_Nmae == HTuple("NVIDIA GeForce RTX 3060 Laptop GPU"))))
            {
                OpenComputeDevice(HTuple(hv_DeviceIdentifiers[hv_i]), &hv_DeviceHandle);
                break;
            }
        }
    }
    if (0 != (int(hv_DeviceHandle != 0))) {
        HTuple hv_func;
        SetComputeDeviceParam(hv_DeviceHandle, "asynchronous_execution", "false");
        hv_func.Clear();
        hv_func[0] = "rgb1_to_gray";
        hv_func[1] = "projective_trans_image";
        hv_func[2] = "find_ncc_models";
        hv_func[3] = "affine_trans_image";
        hv_func[4] = "sub_image";
        hv_func[5] = "hom_mat2d_translate";
        hv_func[6] = "affine_trans_image";
        hv_func[7] = "select_objreduce_domain";
        hv_func[8] = "threshold";
        hv_func[9] = "connection";
        hv_func[10] = "select_shape";
        hv_func[11] = "sub_image";
        hv_func[12] = "dual_threshold";
        hv_func[13] = "tuple_length";
        InitComputeDevice(hv_DeviceHandle, hv_func);
        ActivateComputeDevice(hv_DeviceHandle);
    }
}

hconAction::hconAction()
{
    /*Halcon源码*/
    SPDLOG_INFO("halconAction start!start GPU speed up...");
    //开启GPU加速
    //OpenGPU();
    SPDLOG_INFO("finish GPU speed up");
}

hconAction::~hconAction()
{
    SPDLOG_INFO("halconAction end!");
   //DeactivateComputeDevice(hv_DeviceHandle);  //取消激活GPU
}

void GenPolygonPoints(HObject ho_GrayImage, HTuple hv_ThresholdVal, HTuple* hv_Xl, HTuple* hv_Yl, HTuple* hv_iErrorCode)
{

    // Local iconic variables
    HObject  ho_Region, ho_ConnectedRegions, ho_SelectedRegions;
    HObject  ho_RegionUnion, ho_RegionTrans, ho_Contours, ho_ContoursSplit;
    HObject  ho_SelectedContours, ho_RegressContours, ho_UnionContours;
    HObject  ho_RegionLines;

    // Local control variables
    HTuple  hv_Row1, hv_Column, hv_Phi, hv_Length1;
    HTuple  hv_Length2, hv_RowBegin, hv_ColBegin, hv_RowEnd;
    HTuple  hv_ColEnd, hv_Nr, hv_Nc, hv_Dist, hv_Length, hv_X;
    HTuple  hv_Y, hv_index, hv_Y1Start, hv_X1Start, hv_Y1End;
    HTuple  hv_X1End, hv_Y2Start, hv_X2Start, hv_Y2End, hv_X2End;
    HTuple  hv_Row, hv_Col, hv_IsOverlapping;


    Threshold(ho_GrayImage, &ho_Region, hv_ThresholdVal, 255);
    Connection(ho_Region, &ho_ConnectedRegions);
    SelectShape(ho_ConnectedRegions, &ho_SelectedRegions, "area", "and", 500000, 2e+8);
    Union1(ho_SelectedRegions, &ho_RegionUnion);
    //变换区域形状
    ShapeTrans(ho_RegionUnion, &ho_RegionTrans, "convex");
    SmallestRectangle2(ho_RegionTrans, &hv_Row1, &hv_Column, &hv_Phi, &hv_Length1,
        &hv_Length2);
    //Regon->xld
    GenContourRegionXld(ho_RegionTrans, &ho_Contours, "border");
    //轮廓分割(线)并筛选
    SegmentContoursXld(ho_Contours, &ho_ContoursSplit, "lines", 20, 30, 20);
    SelectContoursXld(ho_ContoursSplit, &ho_SelectedContours, "contour_length", 100,
        20000, -0.5, 0.5);
    //拟合轮廓的直线
    RegressContoursXld(ho_SelectedContours, &ho_RegressContours, "median", 3);
    UnionCollinearContoursXld(ho_RegressContours, &ho_UnionContours, 10, 1, 2, 0.1,
        "attr_keep");
    FitLineContourXld(ho_UnionContours, "tukey", -1, 0, 5, 2, &hv_RowBegin, &hv_ColBegin,
        &hv_RowEnd, &hv_ColEnd, &hv_Nr, &hv_Nc, &hv_Dist);
    GenRegionLine(&ho_RegionLines, hv_RowBegin, hv_ColBegin, hv_RowEnd, hv_ColEnd);
    TupleLength(hv_RowBegin, &hv_Length);
    if (0 != (int(hv_Length != 6)))
    {
        (*hv_iErrorCode) = 101;
        return;
    }
    if (0 != (int((hv_Phi.TupleAbs()) > (HTuple(10).TupleRad()))))
    {
        (*hv_iErrorCode) = 102;
        return;
    }
    //计算交点坐标
    hv_X = HTuple();
    hv_Y = HTuple();
    if (HDevWindowStack::IsOpen())
        DispObj(ho_GrayImage, HDevWindowStack::GetActive());
    for (hv_index = 0; hv_index <= 5; hv_index += 1)
    {
        if (0 != (int(hv_index == 5)))
        {
            //第一条直线起始点
            hv_Y1Start = HTuple(hv_RowBegin[hv_index]);
            hv_X1Start = HTuple(hv_ColBegin[hv_index]);
            hv_Y1End = HTuple(hv_RowEnd[hv_index]);
            hv_X1End = HTuple(hv_ColEnd[hv_index]);
            //第二条直线的起始点
            hv_Y2Start = ((const HTuple&)hv_RowBegin)[0];
            hv_X2Start = ((const HTuple&)hv_ColBegin)[0];
            hv_Y2End = ((const HTuple&)hv_RowEnd)[0];
            hv_X2End = ((const HTuple&)hv_ColEnd)[0];
        }
        else
        {
            //第一条直线起始点
            hv_Y1Start = HTuple(hv_RowBegin[hv_index]);
            hv_X1Start = HTuple(hv_ColBegin[hv_index]);
            hv_Y1End = HTuple(hv_RowEnd[hv_index]);
            hv_X1End = HTuple(hv_ColEnd[hv_index]);
            //第二条直线的起始点
            hv_Y2Start = HTuple(hv_RowBegin[hv_index + 1]);
            hv_X2Start = HTuple(hv_ColBegin[hv_index + 1]);
            hv_Y2End = HTuple(hv_RowEnd[hv_index + 1]);
            hv_X2End = HTuple(hv_ColEnd[hv_index + 1]);
        }
        //计算角点坐标
        IntersectionLines(hv_Y1Start, hv_X1Start, hv_Y1End, hv_X1End, hv_Y2Start, hv_X2Start,
            hv_Y2End, hv_X2End, &hv_Row, &hv_Col, &hv_IsOverlapping);
        hv_X = hv_X.TupleConcat(hv_Col);
        hv_Y = hv_Y.TupleConcat(hv_Row);
    }
    if (0 != (int(hv_ColBegin > hv_Column)))
    {
        (*hv_Xl) = ((const HTuple&)hv_X)[(((((HTuple(5).Append(0)).Append(1)).Append(2)).Append(3)).Append(4))];
        (*hv_Yl) = ((const HTuple&)hv_Y)[(((((HTuple(5).Append(0)).Append(1)).Append(2)).Append(3)).Append(4))];
    }
    else
    {
        (*hv_Xl) = hv_X;
        (*hv_Yl) = hv_Y;
    }
    (*hv_iErrorCode) = 0;
    return;
}

Mat hconAction::HImageToMat(const HalconCpp::HImage& hImg)
{
    cv::Mat mat;
    int channels = hImg.CountChannels()[0].I();
    HalconCpp::HImage hImage = hImg.ConvertImageType("byte");

    Hlong hW = 0, hH = 0;
    HalconCpp::HString cType;

    if (channels == 1) {
        void* r = hImage.GetImagePointer1(&cType, &hW, &hH);
        mat.create(int(hH), int(hW), CV_8UC1);
        memcpy(mat.data, static_cast<unsigned char*>(r), int(hW * hH));
    }
    else if (channels == 3) {
        void* r = NULL, * g = NULL, * b = NULL;

        hImage.GetImagePointer3(&r, &g, &b, &cType, &hW, &hH);
        mat.create(int(hH), int(hW), CV_8UC3);

        std::vector<cv::Mat> vec(3);
        vec[0].create(int(hH), int(hW), CV_8UC1);
        vec[1].create(int(hH), int(hW), CV_8UC1);
        vec[2].create(int(hH), int(hW), CV_8UC1);

        memcpy(vec[2].data, static_cast<unsigned char*>(r), int(hW * hH));
        memcpy(vec[1].data, static_cast<unsigned char*>(g), int(hW * hH));
        memcpy(vec[0].data, static_cast<unsigned char*>(b), int(hW * hH));
        cv::merge(vec, mat);
    }
    return mat;
}

Mat hconAction::HObjectToMat(const HalconCpp::HObject& hObj)
{
    HalconCpp::HImage hImg(hObj);
    return HImageToMat(hImg);
}

HalconCpp::HObject hconAction::MatToHObject(const Mat& image)
{
    HalconCpp::HObject Hobj = HalconCpp::HObject();
    int hgt = image.rows;
    int wid = image.cols;
    int i;
    if (image.type() == CV_8UC3)
    {
        std::vector<Mat> imgchannel;
        split(image, imgchannel);  //1ms
        Mat imgB = imgchannel[0];
        Mat imgG = imgchannel[1];
        Mat imgR = imgchannel[2];
        uchar* dataR = new uchar[hgt * wid];
        uchar* dataG = new uchar[hgt * wid];
        uchar* dataB = new uchar[hgt * wid];
        for (i = 0; i < hgt; i++)   //1-2ms
        {
            memcpy(dataR + wid * i, imgR.data + imgR.step * i, wid);
            memcpy(dataG + wid * i, imgG.data + imgG.step * i, wid);
            memcpy(dataB + wid * i, imgB.data + imgB.step * i, wid);
        }
        HalconCpp::GenImage3(&Hobj, "byte", wid, hgt, (Hlong)dataR, (Hlong)dataG, (Hlong)dataB);  //1ms
        delete[]dataR;
        delete[]dataG;
        delete[]dataB;
        dataR = NULL;
        dataG = NULL;
        dataB = NULL;
    }
    else if (image.type() == CV_8UC1)
    {
        uchar* data = new uchar[hgt * wid];
        for (i = 0; i < hgt; i++)
            memcpy(data + wid * i, image.data + image.step * i, wid);

        HalconCpp::GenImage1(&Hobj, "byte", wid, hgt, (Hlong)data);
        delete[] data;
        data = NULL;
    }
    return Hobj;
}

HalconCpp::HImage hconAction::MatToHImage(const Mat& image)
{
    HalconCpp::HImage hImg(MatToHObject(image));
    return hImg;
}

void hconAction::SaveHobjectImg(string path, HObject& Hobject_in)
{
    Mat out = HObjectToMat(Hobject_in);
    imwrite(path, out);
}

void ChangeCorner(HTuple hv_XM, HTuple hv_YM, HTuple* hv_XMOut, HTuple* hv_YMOut)
{

    // Local iconic variables

    // Local control variables
    HTuple  hv_XM1, hv_XM2, hv_YM1, hv_YM2;

    (*hv_XMOut) = hv_XM;
    (*hv_YMOut) = hv_YM;
    TupleSelectRange((*hv_XMOut), 0, 2, &hv_XM1);
    TupleSelectRange((*hv_XMOut), 5, 5, &hv_XM2);
    TupleConcat(hv_XM1, hv_XM2, &(*hv_XMOut));

    TupleSelectRange((*hv_YMOut), 0, 2, &hv_YM1);
    TupleSelectRange((*hv_YMOut), 5, 5, &hv_YM2);
    TupleConcat(hv_YM1, hv_YM2, &(*hv_YMOut));
    return;
}

void DetectWhiteArea(unsigned int now_num, HObject &ho_ReducedBackground, HObject &ho_RegionIntersection, HObject &ho_LightSelected,
    HObject &ho_WhiteRegion, HObject &ho_BackgroundRegion, HTuple hv_MinArea, HTuple hv_MaxArea,
    HTuple hv_WhiteMin, HTuple hv_WhiteMax)
{
    // Local iconic variables
    HObject  ho_ReducedBackgroundOut, ho_LightRegion, ho_ImageEmphasize1, ho_ImageMean, ho_RegionDynThresh;
    HObject  ho_RegionTrans, ho_RegionErosion, ho_LightImage;
    HObject  ho_LightMean, ho_LightDynThresh, ho_LightConnected;
    HObject  ho_WhiteThreshold, ho_ConnectedRegions;

    // Local control variables
    HTuple  hv_UsedThreshold1;

    if (now_num == 4 || now_num == 6) {
        Emphasize(ho_ReducedBackground, &ho_ImageEmphasize1, 3, 3, 1.5);

        MeanImage(ho_ReducedBackground, &ho_ImageMean, 5, 5);
        DynThreshold(ho_ImageEmphasize1, ho_ImageMean, &ho_RegionDynThresh, 7, "light");
        Connection(ho_RegionDynThresh, &ho_ConnectedRegions);

        SelectShape(ho_ConnectedRegions, &ho_WhiteRegion, "area", "and", 4, 99999);
    }
    else if (now_num == 1) {
        ho_ReducedBackgroundOut = ho_ReducedBackground;
        BinaryThreshold(ho_ReducedBackgroundOut, &ho_LightRegion, "max_separability", "light",
            &hv_UsedThreshold1);
        Connection(ho_LightRegion, &ho_LightRegion);
        SelectShape(ho_LightRegion, &ho_LightRegion, "width", "and", 10, 300);
        
        ShapeTrans(ho_LightRegion, &ho_RegionTrans, "rectangle1");
        ErosionRectangle1(ho_RegionTrans, &ho_RegionErosion, 9, 1);
        ReduceDomain(ho_ReducedBackgroundOut, ho_RegionErosion, &ho_LightImage);
        MeanImage(ho_LightImage, &ho_LightMean, 9, 9);
        DynThreshold(ho_LightImage, ho_LightMean, &ho_LightDynThresh, 5, "dark");
        Connection(ho_LightDynThresh, &ho_LightConnected);
        SelectShape(ho_LightConnected, &ho_LightSelected, "area", "and", hv_MinArea,
            hv_MaxArea);

        DilationRectangle1(ho_LightRegion, &ho_LightRegion, 7, 1);
        Difference(ho_RegionIntersection, ho_LightRegion, &ho_BackgroundRegion);
        ReduceDomain(ho_ReducedBackgroundOut, ho_BackgroundRegion, &ho_ReducedBackgroundOut);

        Threshold(ho_ReducedBackgroundOut, &ho_WhiteThreshold, hv_WhiteMin, hv_WhiteMax);

        Connection(ho_WhiteThreshold, &ho_ConnectedRegions);

        SelectShape(ho_ConnectedRegions, &ho_WhiteRegion, (HTuple("area").Append("width")),
            "and", hv_MinArea.TupleConcat(3), hv_MaxArea.TupleConcat(10));
    }
    else {
        //修改部分 检测条纹区白色渣点
        Threshold(ho_ReducedBackground, &ho_WhiteThreshold, hv_WhiteMin, hv_WhiteMax);

        Connection(ho_WhiteThreshold, &ho_ConnectedRegions);

        SelectShape(ho_ConnectedRegions, &ho_WhiteRegion, (HTuple("area").Append("width")),
            "and", hv_MinArea.TupleConcat(3), hv_MaxArea.TupleConcat(10));
    }
}

// Short Description: 根据模板参数，在不同的区域以不同的参数值检测刀线 
void knife_line_detect(HObject &ho_ReducedBackground, HObject& ho_KnifeRegions, HTuple hv_cannyAlpha,
    HTuple hv_cannyLow, HTuple hv_cannyHigh, HTuple hv_adjacentDisAbs, HTuple hv_adjacentDisRel,
    HTuple hv_contourMin, HTuple hv_contourMax)
{

    // Local iconic variables
    HObject  ho_LineEdges, ho_VerticalLines, ho_FixedLineEdges;
    HObject  ho_ValidLines, ho_ValidLinesRegion;

    EdgesSubPix(ho_ReducedBackground, &ho_LineEdges, "canny", hv_cannyAlpha, hv_cannyLow,
        hv_cannyHigh);

    SelectContoursXld(ho_LineEdges, &ho_VerticalLines, "direction", HTuple(80).TupleRad(),
        HTuple(100).TupleRad(), -0.5, 0.5);

    UnionAdjacentContoursXld(ho_VerticalLines, &ho_FixedLineEdges, hv_adjacentDisAbs,
        hv_adjacentDisRel, "attr_keep");
    SelectContoursXld(ho_FixedLineEdges, &ho_ValidLines, "contour_length", hv_contourMin,
        hv_contourMax, -0.5, 0.5);

    GenRegionContourXld(ho_ValidLines, &ho_ValidLinesRegion, "filled");
    Connection(ho_ValidLinesRegion, &ho_KnifeRegions);
    return;
}

int hconAction::GetIni(string file_path, string apname, string partname)
{
    int num = 0;
    char getchr[10];	//接收配置文件字符的缓冲区
    DWORD res;		//返回值
    file_path += "config.ini";
    res = GetPrivateProfileString(apname.c_str(), partname.c_str(), "", getchr, sizeof(getchr), file_path.c_str());

    if (res > 0) {//说明该参数存在
        num = atoi(getchr);
    }
    else {
        num = -1;
    }

    //std::cout << "GetIni-------file_path:" << file_path << ",apaname:" << apname <<",partname:" << partname << ",get res:" << to_string(res) << ",get num =" << to_string(num) << std::endl;

    return num;
}

//往配置文件里写数据
int hconAction::SetIni(string file_path, string apname, string partname, int set_num)
{
    //例：WritePrivateProfileString("LiMing", "Sex", "Man", lpPath);
    string num_str = to_string(set_num);
    WritePrivateProfileString(apname.c_str(), partname.c_str(), num_str.c_str(), file_path.c_str());
}

void SendResultToVector(int class_res, int x1, int y1, int x2, int y2)
{
    Detect_Result result;
    result.class_res = class_res;

    result.y1 = y1;
    result.x1 = x1;
    result.y2 = y2;
    result.x2 = x2;
    //result.result_str = "{\"class\":" + to_string(result.class_res) + ",\"config\":99" +
    //    +",\"ix1\":" + to_string(result.x1) + ",\"iy1\":" + to_string(result.y1)
     //   + ",\"ix2\":" + to_string(result.x2) + ",\"iy2\":" + to_string(result.y2) + "}";

    result.result_str = "{\"class\":" + to_string(result.class_res) + "}";

    halcon_result_mutex.lock();
    Halcon_Result.push_back(result);
    halcon_result_mutex.unlock();
}

//更新模板数据
void hconAction::HALCON_Refresh_Model_all(string filePath_moban)
{
    int ini_result = 0;
    int moban_num = 1;

    Halcon_moban_all.clear();
    SPDLOG_INFO("HALCON Refresh_Model_all start!moban size:{},path:{}", Halcon_moban_all.size(), filePath_moban);

    //创建二维码检测模型
    CreateDataCode2dModel("QR Code", HTuple(), HTuple(), &hv_DataCodeHandle);
    //设置相关参数
    SetDataCode2dParam(hv_DataCodeHandle, "polarity", "dark_on_light");
    SetDataCode2dParam(hv_DataCodeHandle, "module_gap", "no");
    SetDataCode2dParam(hv_DataCodeHandle, "mirrored", "no");
    SetDataCode2dParam(hv_DataCodeHandle, "timeout", 10);
    SetDataCode2dParam(hv_DataCodeHandle, "small_modules_robustness", "high");

    /*模板图像*/
    string ImageModel_path = filePath_moban + "0.bmp";
    HObject ho_ImageModel;
    ReadImage(&ho_ImageModel, ImageModel_path.c_str());

    HalconCpp::HObject ho_ImageGrayM;
    HalconCpp::HTuple hv_iErrorCode;
    Rgb1ToGray(ho_ImageModel, &ho_ImageGrayM);

    calculate.ThresholdVal = GetIni(filePath_moban, "common", "ThresholdVal");

    HalconCpp::HTuple hv_ThresholdVal(calculate.ThresholdVal);
    GenPolygonPoints(ho_ImageGrayM, hv_ThresholdVal, &calculate.hv_XM, &calculate.hv_YM, &hv_iErrorCode);

    ChangeCorner(calculate.hv_XM, calculate.hv_YM, &calculate.hv_XM, &calculate.hv_YM);

    while (1) {
        hconAction_moban moban_s;
        string mobanname = "moban" + to_string(moban_num);
        moban_s.x1 = GetIni(filePath_moban, mobanname, "x1");
        if (moban_s.x1 == -1) break;   //配置文件无该模板的配置
        moban_s.y1 = GetIni(filePath_moban, mobanname, "y1");
        moban_s.x2 = GetIni(filePath_moban, mobanname, "x2");
        moban_s.y2 = GetIni(filePath_moban, mobanname, "y2");

        moban_s.BlackMin = GetIni(filePath_moban, mobanname, "BlackMin");
        moban_s.WhiteMin = GetIni(filePath_moban, mobanname, "WhiteMin");
        moban_s.BlackMax = GetIni(filePath_moban, mobanname, "BlackMax");
        moban_s.WhiteMax = GetIni(filePath_moban, mobanname, "WhiteMax");

        moban_s.RatioDila = (float)GetIni(filePath_moban, mobanname, "RatioDila") / 10.0;
        moban_s.DiffThr = GetIni(filePath_moban, mobanname, "DiffThr") / 10.0;
        moban_s.DiffThrD = (float)GetIni(filePath_moban, mobanname, "DiffThrD") / 10.0;
        moban_s.MinArea = GetIni(filePath_moban, mobanname, "MinArea");
        moban_s.MaxArea = GetIni(filePath_moban, mobanname, "MaxArea");

        moban_s.XOffSet = GetIni(filePath_moban, mobanname, "XOffSet");
        moban_s.YOffSet = GetIni(filePath_moban, mobanname, "YOffSet");

        moban_s.Mult = GetIni(filePath_moban, mobanname, "Mult");
        moban_s.Add = GetIni(filePath_moban, mobanname, "Add");

        moban_s.cannyAlpha = (float)GetIni(filePath_moban, mobanname, "cannyAlpha")/10.0;
        moban_s.cannyLow = (float)GetIni(filePath_moban, mobanname, "cannyLow") / 10.0;
        moban_s.cannyHigh = (float)GetIni(filePath_moban, mobanname, "cannyHigh") / 10.0;
        moban_s.adjacentDisAbs = (float)GetIni(filePath_moban, mobanname, "adjacentDisAbs") / 10.0;
        moban_s.adjacentDisRel = (float)GetIni(filePath_moban, mobanname, "adjacentDisRel") / 10.0;
        moban_s.contourMin = GetIni(filePath_moban, mobanname, "contourMin");
        moban_s.contourMax = GetIni(filePath_moban, mobanname, "contourMax");

        moban_s.now_num = moban_num;

        /*模板坐标：Row(y),Col(x)*/
        HalconCpp::HTuple Row1(moban_s.y1);
        HalconCpp::HTuple Col1(moban_s.x1);
        HalconCpp::HTuple Row2(moban_s.y2);
        HalconCpp::HTuple Col2(moban_s.x2);

        //阈值分割参数：Black与White分别代表黑点白点
        HalconCpp::HTuple hv_BlackMin(moban_s.BlackMin);
        HalconCpp::HTuple hv_WhiteMin(moban_s.WhiteMin);
        HalconCpp::HTuple hv_BlackMax(moban_s.BlackMax);
        HalconCpp::HTuple hv_WhiteMax(moban_s.WhiteMax);
        HalconCpp::HTuple hv_RatioDila(moban_s.RatioDila);


        HalconCpp::HObject ho_Region11, ho_ConnectedRegions, ho_SelectedRegions, ho_Region3;
        HalconCpp::HTuple hv_AreaModel, hv_RowModel, hv_ColumnModel;

        //生成背景Mask
        GenRectangle1(&moban_s.mid_result.ho_ModelRegion, Row1, Col1, Row2, Col2);
        ReduceDomain(ho_ImageGrayM, moban_s.mid_result.ho_ModelRegion, &moban_s.mid_result.ho_TemplateImageM);
        Threshold(moban_s.mid_result.ho_TemplateImageM, &moban_s.mid_result.ho_RegionM, hv_BlackMin, hv_BlackMax);

        if (moban_num == 11){
            Threshold(moban_s.mid_result.ho_TemplateImageM, &ho_Region11, 166, 255);
            DilationCircle(ho_Region11, &ho_Region11, 5);
            Union2(ho_Region11, moban_s.mid_result.ho_RegionM, &moban_s.mid_result.ho_RegionM);
        }
        //将模板3的人像抠出来
        if (moban_num == 3) {
            Connection(moban_s.mid_result.ho_RegionM, &ho_ConnectedRegions);
            SelectShape(ho_ConnectedRegions, &ho_SelectedRegions, "ra", "and", 100, 110);
            ShapeTrans(ho_SelectedRegions, &ho_Region3, "convex");
            Union2(moban_s.mid_result.ho_RegionM, ho_Region3, &moban_s.mid_result.ho_RegionM);
        }

        DilationCircle(moban_s.mid_result.ho_RegionM, &moban_s.mid_result.ho_RegionDilationM, hv_RatioDila);
        Difference(moban_s.mid_result.ho_ModelRegion, moban_s.mid_result.ho_RegionDilationM, &moban_s.mid_result.ho_RegionIntersection);
        //计算模板的连通域面积
        AreaCenter(moban_s.mid_result.ho_RegionM, &hv_AreaModel, &hv_RowModel, &hv_ColumnModel);
        //创建ncc模版
        CreateNccModel(moban_s.mid_result.ho_TemplateImageM, "auto", -0.1, 0.2, "auto", "use_polarity", &moban_s.mid_result.hv_ModelID);
        //模版中心参考坐标
        moban_s.mid_result.hv_RefRow = (Row1 + Row2) / 2;
        moban_s.mid_result.hv_RefCol = (Col1 + Col2) / 2;

        //std::cout << "HALCON Refresh_Model_all moban:" << to_string(moban_s.now_num) << ",y1:" << to_string(moban_s.y1) << ",x1:" << to_string(moban_s.x1) << ",RatioDila:" << to_string(hv_RatioDila[0].D())<<
        //    ",hv_Row11:" << to_string(moban_s.mid_result.hv_Row11[0].D()) << ",hv_Column11:" << to_string(moban_s.mid_result.hv_Column11[0].D()) << ",hv_Row2:" << to_string(moban_s.mid_result.hv_Row2[0].D()) << ",hv_Column2:" << to_string(moban_s.mid_result.hv_Column2[0].D())
        //<< std::endl;

        SPDLOG_INFO("HALCON Refresh_Model_all push moban now_num:{:d},y1:{:d},x1:{:d},RatioDila:{:f},DiffThrD:{:f}",
            moban_s.now_num, moban_s.y1, moban_s.x1, moban_s.RatioDila, moban_s.DiffThrD);

        moban_num++;
        Halcon_moban_all.push_back(moban_s);
    }

    //std::cout << "HALCON Refresh_Model_all finish!moban size:" << to_string(Halcon_moban_all.size()) << std::endl;
    SPDLOG_INFO("HALCON Refresh_Model_all finish!moban size:{}", Halcon_moban_all.size());
}

//样品图片中间的变换过程
int hconAction::ImageChange(HalconCpp::HObject& ho_ImageInput, HalconCpp::HObject& ho_TransImage)
{
    //std::clock_t t1, t2;
    //t1 = clock();

    //OpenGPU();

    HalconCpp::HObject ho_GrayImage;
    HalconCpp::HTuple hv_Xl, hv_Yl, hv_iErrorCode, hv_HomMat2D;

    HalconCpp::HTuple hv_ThresholdVal(calculate.ThresholdVal);

    Rgb1ToGray(ho_ImageInput, &ho_GrayImage);
    GenPolygonPoints(ho_GrayImage, hv_ThresholdVal, &hv_Xl, &hv_Yl, &hv_iErrorCode);
    if (0 != (int(hv_iErrorCode != 0))) {   //角点获取失败
        return -1;
    }
    
    ChangeCorner(hv_Xl, hv_Yl, &hv_Xl, &hv_Yl);

    //投影变换
    HomVectorToProjHomMat2d(hv_Yl, hv_Xl, (((HTuple(1).Append(1)).Append(1)).Append(1)), 
        calculate.hv_YM, calculate.hv_XM, (((HTuple(1).Append(1)).Append(1)).Append(1)), "normalized_dlt",
        &hv_HomMat2D);
    ProjectiveTransImage(ho_GrayImage, &ho_TransImage, hv_HomMat2D, "bilinear", "false",
        "false");

    //SaveHobjectImg("D:/OpenCV_test/OpenCV/out/build/x64-Release/Testing/result-moban/" + to_string(now_dealImg_num) + "_ProjectiveTransImage.bmp", ho_TransImage);
    //t2 = clock();

    //SPDLOG_INFO("HALCON ImageChange img:{:d},finish!usetime:{:d}", now_dealImg_num, t2 - t1);
    return 1;
}

void ImageThreadRun_camera2(int threadnum, hconAction_calculate& calculate, hconAction_moban& Model, HalconCpp::HObject& ho_TransImage, unsigned int now_img_num)
{
    std::clock_t t1, t2, t3, t4;
    t1 = clock();

    //阈值分割参数：Black与White分别代表黑点白点
    HalconCpp::HTuple hv_BlackMin(Model.BlackMin);
    HalconCpp::HTuple hv_WhiteMin(Model.WhiteMin);
    HalconCpp::HTuple hv_BlackMax(Model.BlackMax);
    HalconCpp::HTuple hv_WhiteMax(Model.WhiteMax);

    HalconCpp::HTuple hv_MinArea(Model.MinArea);
    HalconCpp::HTuple hv_MaxArea(Model.MaxArea);

    HalconCpp::HTuple hv_DiffThr(Model.DiffThr);

    HalconCpp::HTuple hv_cannyAlpha(Model.cannyAlpha);
    HalconCpp::HTuple hv_cannyLow(Model.cannyLow);
    HalconCpp::HTuple hv_cannyHigh(Model.cannyHigh);
    HalconCpp::HTuple hv_adjacentDisAbs(Model.adjacentDisAbs);
    HalconCpp::HTuple hv_adjacentDisRel(Model.adjacentDisRel);
    HalconCpp::HTuple hv_contourMin(Model.contourMin);
    HalconCpp::HTuple hv_contourMax(Model.contourMax);

    HalconCpp::HTuple hv_Flag, hv_Row, hv_Column, hv_Angle, hv_Score;
    HalconCpp::HTuple hv_HomMat2DIdentity, hv_delta_X, hv_delta_Y, hv_HomMat2DTranslate;
    HalconCpp::HObject ho_ImageAffineTrans, ho_ReducedBackground, ho_BinaryBackground, ho_ConnectedBackground, ho_SelectedRegions;
    HalconCpp::HTuple  hv_Row11, hv_Column11, hv_Row2, hv_Column2;

    HalconCpp::HObject ho_ImageSub, ho_WhiteRegion, ho_BlackRegion, ho_RegionUnion, ho_ReducedBackground_1;
    HalconCpp::HTuple hv_WhiteArea, hv_WhiteRow, hv_WhiteCol, hv_WhiteNum;
    HalconCpp::HTuple hv_BlackArea, hv_BlackRow, hv_BlackCol, hv_BlackNum;
    HalconCpp::HTuple hv_KnifeArea, hv_KnifeRow, hv_KnifeCol, hv_KnifeNum;

    HalconCpp::HObject ho_TextRegion, ho_ImageEmphasize, ho_TextRegionDilation, ho_RegionMD;
    HalconCpp::HObject ho_TextRegionDifference, ho_TextImageReduced, ho_TextRegions;
    HalconCpp::HTuple hv_UsedThreshold, hv_TextArea, hv_TextRow, hv_TextColumn;
    HalconCpp::HTuple hv_TextRow1, hv_TextColumn1, hv_TextRow2, hv_TextColumn2;
    HalconCpp::HObject ho_ImageMean, ho_RegionDynThresh, ho_ConnectedRegions;

    HalconCpp::HObject ho_LightSelected, ho_BackgroundRegion, ho_ImageEmphasize2;
    HalconCpp::HTuple hv_LightArea, hv_LightRow, hv_LightCol, hv_LightNum;

    hv_Flag = 0;

    //定位
    //t3 = clock();
    FindNccModel(ho_TransImage, Model.mid_result.hv_ModelID, -0.1, 0.2, 0.3, 1, 0.5, "true", 0, &hv_Row,
        &hv_Column, &hv_Angle, &hv_Score);
    //t4 = clock();

    if (0 != (int(hv_Score < hv_DiffThr))) {
        hv_Flag = 2;
        goto detectfail;
    }

    //仿射变换校正
    HomMat2dIdentity(&hv_HomMat2DIdentity);
    hv_delta_X = Model.mid_result.hv_RefCol - hv_Column;
    hv_delta_Y = Model.mid_result.hv_RefRow - hv_Row;
    HomMat2dTranslate(hv_HomMat2DIdentity, hv_delta_Y, hv_delta_X, &hv_HomMat2DTranslate);
    AffineTransImage(ho_TransImage, &ho_ImageAffineTrans, hv_HomMat2DTranslate, "constant",
        "false");
    //前景扣除
    ReduceDomain(ho_ImageAffineTrans, Model.mid_result.ho_RegionIntersection, &ho_ReducedBackground);

    //------------------0.文字区域缺陷检测-------------------------------------------------
    if (Model.now_num == 2) {//只对区域2做检测   耗时：1-2ms
        //t3 = clock();
        ReduceDomain(ho_ImageAffineTrans, Model.mid_result.ho_RegionDilationM, &ho_TextRegion);
        Emphasize(ho_TextRegion, &ho_ImageEmphasize, 7, 7, 1.8);
        BinaryThreshold(ho_ImageEmphasize, &ho_TextRegion, "max_separability", "dark", &hv_UsedThreshold);
        DilationRectangle1(ho_TextRegion, &ho_TextRegionDilation, 3, 3);
        DilationRectangle1(Model.mid_result.ho_RegionM, &ho_RegionMD, 5, 5);
        Difference(Model.mid_result.ho_RegionDilationM, ho_TextRegionDilation, &ho_TextRegionDifference);
        ReduceDomain(ho_ImageEmphasize, ho_TextRegionDifference, &ho_TextImageReduced);
        Threshold(ho_TextImageReduced, &ho_TextRegions, 0, 116);
        AreaCenter(ho_TextRegions, &hv_TextArea, &hv_TextRow, &hv_TextColumn);

        if (0 != (int(hv_TextArea > 5))) {
            hv_Flag = 1;
            SmallestRectangle1(ho_TextRegions, &hv_TextRow1, &hv_TextColumn1, &hv_TextRow2, &hv_TextColumn2);
        }
        //t4 = clock();
        //std::cout << "TextArea usetime:" << to_string(t4 - t3) << endl;
    }

    //--------------------------------------1.背景白色缺陷检测-------------------------------------------------
    DetectWhiteArea(Model.now_num, ho_ReducedBackground, Model.mid_result.ho_RegionIntersection, ho_LightSelected, ho_WhiteRegion,
                        ho_BackgroundRegion, hv_MinArea, hv_MaxArea, hv_WhiteMin, hv_WhiteMax);

    AreaCenter(ho_WhiteRegion, &hv_WhiteArea, &hv_WhiteRow, &hv_WhiteCol);
    if (Model.now_num == 1)
        AreaCenter(ho_LightSelected, &hv_LightArea, &hv_LightRow, &hv_LightCol);

    TupleLength(hv_LightArea, &hv_LightNum);
    TupleLength(hv_WhiteArea, &hv_WhiteNum);

    //--------------------------------------2.背景黑色缺陷检测------------------------------------------------
    //t3 = clock();
    if (Model.now_num == 4 || Model.now_num == 6) {
        MeanImage(ho_ReducedBackground, &ho_ImageMean, 9, 9);
        DynThreshold(ho_ReducedBackground, ho_ImageMean, &ho_RegionDynThresh, 5, "dark");
        Connection(ho_RegionDynThresh, &ho_ConnectedBackground);
    }
    else if (Model.now_num == 1 || Model.now_num == 3 || Model.now_num == 7) {
        Emphasize(ho_ReducedBackground, &ho_ImageEmphasize2, 9, 9, 4);
        if (Model.now_num == 1){
            ReduceDomain(ho_ImageEmphasize2, ho_BackgroundRegion, &ho_ImageEmphasize2);
        }
        Threshold(ho_ImageEmphasize2, &ho_BinaryBackground, 0, 13);
        Connection(ho_BinaryBackground, &ho_ConnectedBackground);
        SelectShape(ho_ConnectedBackground, &ho_ConnectedBackground, (HTuple("width").Append("anisometry")), "and", (HTuple(3).Append(1)), (HTuple(100).Append(10)));
    }
    else {
        Threshold(ho_ReducedBackground, &ho_BinaryBackground, hv_BlackMin, hv_BlackMax);
        Connection(ho_BinaryBackground, &ho_ConnectedBackground);
    }
    SelectShape(ho_ConnectedBackground, &ho_BlackRegion, "area", "and", hv_MinArea, hv_MaxArea);
    AreaCenter(ho_BlackRegion, &hv_BlackArea, &hv_BlackRow, &hv_BlackCol);

    //TupleLength(hv_BlackArea, &hv_BlackNum);
    //t4 = clock();
    //std::cout << "BlackArea usetime:" << to_string(t4 - t3) << endl;
    
    //--------------------------------------3.刀丝缺陷检测------------------------------------------------
    if ( Model.now_num == 6 ) {
        t3 = clock();
        knife_line_detect(ho_ReducedBackground, ho_ConnectedBackground, hv_cannyAlpha,
            hv_cannyLow, hv_cannyHigh, hv_adjacentDisAbs, hv_adjacentDisRel, hv_contourMin,
            hv_contourMax);
        AreaCenter(ho_ConnectedBackground, &hv_KnifeArea, &hv_KnifeRow, &hv_KnifeCol);
        TupleConcat(hv_BlackArea, hv_KnifeArea, &hv_BlackArea);
        TupleConcat(hv_KnifeRow, hv_BlackRow,&hv_BlackRow);
        TupleConcat(hv_KnifeCol, hv_BlackCol, &hv_BlackCol);
        t4 = clock();
        std::cout << "knife_line_detect usetime:" << to_string(t4 - t3) << endl;
    }

    TupleLength(hv_BlackArea, &hv_BlackNum);
    //--------------------------------------4.结果统计------------------------------------------------
    if (0 != (int(hv_WhiteNum > 0))) {
        HalconCpp::HTuple detect_lens_1;
        HalconCpp::TupleLength(hv_WhiteRow, &detect_lens_1);
        if (detect_lens_1.I() > 3) detect_lens_1 = 3;
        for (int detect_num = 0; detect_num < detect_lens_1.I(); detect_num++) {
            SendResultToVector(1, hv_WhiteCol[detect_num].D() - 5 + hv_delta_X.D(), hv_WhiteRow[detect_num].D() - 5 + hv_delta_Y.D(),
                hv_WhiteCol[detect_num].D() + 5 + hv_delta_X.D(), hv_WhiteRow[detect_num].D() + 5 + hv_delta_Y.D());
        }
    }
    if (0 != (int(hv_LightNum > 0))) {
        HalconCpp::HTuple detect_lens_3;
        HalconCpp::TupleLength(hv_LightRow, &detect_lens_3);
        if (detect_lens_3.I() > 5) detect_lens_3 = 5;
        for (int detect_num = 0; detect_num < detect_lens_3.I(); detect_num++) {
            SendResultToVector(1, hv_LightCol[detect_num].D() - 5 + hv_delta_X.D(), hv_LightRow[detect_num].D() - 5 + hv_delta_Y.D(),
                hv_LightCol[detect_num].D() + 5 + hv_delta_X.D(), hv_LightRow[detect_num].D() + 5 + hv_delta_Y.D());
        }
    }
    if (0 != (int(hv_BlackNum > 0))) {
        HalconCpp::HTuple detect_lens_2;
        HalconCpp::TupleLength(hv_BlackRow, &detect_lens_2);
        if (detect_lens_2.I() > 5) detect_lens_2 = 5;
        for (int detect_num = 0; detect_num < detect_lens_2.I(); detect_num++) {
            SendResultToVector(2, hv_BlackCol[detect_num].D() - 5 + hv_delta_X.D(), hv_BlackRow[detect_num].D() - 5 + hv_delta_Y.D(),
                hv_BlackCol[detect_num].D() + 5 + hv_delta_X.D(), hv_BlackRow[detect_num].D() + 5 + hv_delta_Y.D());
        }
    }
    //形状定位失败
detectfail:

    t2 = clock();

    if (hv_Flag.I() == 1) {   //有缺陷
        HalconCpp::HTuple detect_lens;
        HalconCpp::TupleLength(hv_TextRow1, &detect_lens);
        if (detect_lens.I() > 3) detect_lens = 3;
        for (int detect_num = 0; detect_num < detect_lens; detect_num++)  //文字缺陷
        {
            SendResultToVector(3, hv_TextColumn1[detect_num].D() - 5 + hv_delta_X.D(), hv_TextRow1[detect_num].D() - 5 + hv_delta_Y.D(),
                hv_TextColumn2[detect_num].D() + 5 + hv_delta_X.D(), hv_TextRow2[detect_num].D() + 5 + hv_delta_Y.D());
        }
    }
    else if (hv_Flag.I() == 2) {   //形状定位失败
        SendResultToVector(4, Model.x1, Model.y1, Model.x2, Model.y2);
    }

    //std::cout << "all usetime:" << to_string(t2 - t1) << ",FindNccModel usetime:" << to_string(t4 - t3) << endl;
    if (hv_Flag.I() == 1)
        SPDLOG_INFO("HALCON thread:{:d},img:{:d},moban:{:d},flag:{:d},delta_X:{:f},delta_Y:{:f},usetime:{:d}",
            threadnum, now_img_num, Model.now_num, hv_Flag.I(), hv_delta_X.D(), hv_delta_Y.D(), t2 - t1);
}

//单边检测线程
// Short Description: 本函数基于外轮廓线交点的思路，对印刷偏移进行测量
void detect_deviation(HObject &ho_ImageAffineGray)
{
    std::clock_t t1, t2;
    t1 = clock();

    // Local iconic variables
    HObject  ho_ItemRegion, ho_ItemFillUp, ho_ItemConnected;
    HObject  ho_MaxItem, ho_MaxItemClosing, ho_OuterContours;
    HObject  ho_MaxItemErosion, ho_TargetItem, ho_TargetItemRegions;
    HObject  ho_TargetItemFillUp, ho_TargetItemConnected, ho_Inner;
    HObject  ho_InnerContour, ho_ReferenceLine, ho_ReferenceLineContour;

    // Local control variables
    HTuple  hv_OuterRow, hv_OuterColumn, hv_OuterPhi;
    HTuple  hv_OuterWidth, hv_OuterHeight, hv_InnerRow, hv_InnerColumn;
    HTuple  hv_InnerPhi, hv_InnerWidth, hv_InnerHeight, hv_Width;
    HTuple  hv_Height, hv_OuterRows, hv_OuterColumns, hv_OuterIsOverlapping;
    HTuple  hv_InnerRows, hv_InnerIsOverlapping, hv_Index, hv_OuterRows2BeginIndex;
    HTuple  hv_OuterRows1, hv_OuterRows2, hv_InnerRows2BeginIndex;
    HTuple  hv_InnerRows1, hv_InnerRows2;

    //--------------------计算距离,判断偏移-----------------
    //外轮廓获取
    Threshold(ho_ImageAffineGray, &ho_ItemRegion, 30, 255);
    FillUp(ho_ItemRegion, &ho_ItemFillUp);
    Connection(ho_ItemFillUp, &ho_ItemConnected);


    SelectShape(ho_ItemConnected, &ho_MaxItem, "area", "and", 996546, 2e+06);

    //area_center (ItemConnected, Area, Row1, Column1)
    //*     tuple_sort_index(-Area, Indices)
    //*     select_obj(ItemConnected, MaxItem, Indices[0]+1)

    ClosingCircle(ho_MaxItem, &ho_MaxItemClosing, 5.5);
    GenContourRegionXld(ho_MaxItemClosing, &ho_OuterContours, "border");
    SmallestRectangle2(ho_MaxItemClosing, &hv_OuterRow, &hv_OuterColumn, &hv_OuterPhi,
        &hv_OuterWidth, &hv_OuterHeight);

    //内轮廓获取
    ErosionCircle(ho_MaxItemClosing, &ho_MaxItemErosion, 20);
    ReduceDomain(ho_ImageAffineGray, ho_MaxItemErosion, &ho_TargetItem);

    Threshold(ho_TargetItem, &ho_TargetItemRegions, 4, 104);
    FillUp(ho_TargetItemRegions, &ho_TargetItemFillUp);
    Connection(ho_TargetItemFillUp, &ho_TargetItemConnected);
    SelectShape(ho_TargetItemConnected, &ho_Inner, "area", "and", 119603, 250864);
    SmallestRectangle2(ho_Inner, &hv_InnerRow, &hv_InnerColumn, &hv_InnerPhi, &hv_InnerWidth,
        &hv_InnerHeight);

    //gen_rectangle2_contour_xld (InnerRec, InnerRow, InnerColumn, InnerPhi, InnerWidth, InnerHeight)
    GenContourRegionXld(ho_Inner, &ho_InnerContour, "border");
    GetImageSize(ho_TargetItem, &hv_Width, &hv_Height);

    //获取参考线
    GenRegionLine(&ho_ReferenceLine, 0, hv_InnerColumn, hv_Height, hv_InnerColumn);
    GenContourRegionXld(ho_ReferenceLine, &ho_ReferenceLineContour, "border");
    //获取交点
    IntersectionContoursXld(ho_ReferenceLineContour, ho_OuterContours, "mutual", &hv_OuterRows,
        &hv_OuterColumns, &hv_OuterIsOverlapping);
    IntersectionContoursXld(ho_ReferenceLineContour, ho_InnerContour, "mutual", &hv_InnerRows,
        &hv_InnerColumn, &hv_InnerIsOverlapping);

    TupleSort(hv_OuterRows, &hv_OuterRows);
    TupleSort(hv_InnerRows, &hv_InnerRows);

    {
        HTuple end_val41 = (hv_OuterRows.TupleLength()) - 1;
        HTuple step_val41 = 1;
        for (hv_Index = 1; hv_Index.Continue(end_val41, step_val41); hv_Index += step_val41)
        {
            if (0 != (int(((HTuple(hv_OuterRows[0]) - HTuple(hv_OuterRows[hv_Index])).TupleAbs()) > 100)))
            {
                hv_OuterRows2BeginIndex = hv_Index;
                break;
            }
        }
    }
    TupleSelectRange(hv_OuterRows, 0, hv_OuterRows2BeginIndex - 1, &hv_OuterRows1);
    TupleSelectRange(hv_OuterRows, hv_OuterRows2BeginIndex, (hv_OuterRows.TupleLength()) - 1,
        &hv_OuterRows2);

    {
        HTuple end_val50 = (hv_InnerRows.TupleLength()) - 1;
        HTuple step_val50 = 1;
        for (hv_Index = 1; hv_Index.Continue(end_val50, step_val50); hv_Index += step_val50)
        {
            if (0 != (int(((HTuple(hv_InnerRows[0]) - HTuple(hv_InnerRows[hv_Index])).TupleAbs()) > 100)))
            {
                hv_InnerRows2BeginIndex = hv_Index;
                break;
            }
        }
    }
    TupleSelectRange(hv_InnerRows, 0, hv_InnerRows2BeginIndex - 1, &hv_InnerRows1);
    TupleSelectRange(hv_InnerRows, hv_InnerRows2BeginIndex, (hv_InnerRows.TupleLength()) - 1,
        &hv_InnerRows2);

    hv_Deviation = HTuple(((hv_InnerRows1.TupleMean()) - (hv_OuterRows1.TupleMean())) - ((hv_OuterRows2.TupleMean()) - (hv_InnerRows2.TupleMean()))).TupleAbs();

    t2 = clock();
    std::cout << "detect_deviation all usetime:" << to_string(t2 - t1) << ",hv_Deviation:" << to_string(hv_Deviation.D()) << endl;
    SPDLOG_INFO("HALCON detect_deviation Deviation:{:.2f},usetime:{:d}", hv_Deviation.D(), t2 - t1);
}

//------------------二维码检测-------------------------------------------------
void qr_detect(HObject &ho_ImageAffineTransGray, HObject &ho_ModelRegion)
{
    std::clock_t t1, t2;
    t1 = clock();
    // Local iconic variables
    HObject  ho_ImageReduced, ho_QRMean, ho_QRDynThresh;
    HObject  ho_RegionFillUp, ho_QRConnected, ho_QRRegions, ho_QRReduced;
    HObject  ho_QRCrop, ho_QREmphasize, ho_SymbolXLDs, ho_QR;
    HTuple hv_DecodedDataStrings;
    // Local control variables
    HTuple  hv_ResultHandles, hv_UsedThreshold2, hv_Area1;
    HTuple  hv_Row1, hv_Col1, hv_Area2, hv_Row2, hv_Col2;

    ReduceDomain(ho_ImageAffineTransGray, ho_ModelRegion, &ho_ImageReduced);
    MeanImage(ho_ImageReduced, &ho_QRMean, 30, 30);
    DynThreshold(ho_ImageReduced, ho_QRMean, &ho_QRDynThresh, 5, "light");
    FillUp(ho_QRDynThresh, &ho_RegionFillUp);
    Connection(ho_RegionFillUp, &ho_QRConnected);

    SelectShape(ho_QRConnected, &ho_QRRegions, "width", "and", 60, 130);

    ShapeTrans(ho_QRRegions, &ho_QRRegions, "rectangle1");
    ReduceDomain(ho_ImageReduced, ho_QRRegions, &ho_QRReduced);
    CropDomain(ho_QRReduced, &ho_QRCrop);

    Emphasize(ho_QRCrop, &ho_QREmphasize, 3, 3, 10);

    //检测并解码二维码
    FindDataCode2d(ho_QREmphasize, &ho_SymbolXLDs, hv_DataCodeHandle, "train", "all",
        &hv_ResultHandles, &hv_DecodedDataStrings);
    //DecodedDataStrings是二维码内容
    qr_code_str = hv_DecodedDataStrings.ToString();

    BinaryThreshold(ho_QRReduced, &ho_QR, "max_separability", "dark", &hv_UsedThreshold2);
    ShapeTrans(ho_QR, &ho_QR, "rectangle1");

    HTuple qr_number;
    CountObj(ho_QR, &qr_number);
    //std::cout << "qr_number:" << qr_number.I() << endl;
    if (qr_number.I() == 0) {
        return;
    }

    AreaCenter(ho_QRRegions, &hv_Area1, &hv_Row1, &hv_Col1);
    AreaCenter(ho_QR, &hv_Area2, &hv_Row2, &hv_Col2);

    hv_qr_xx = (hv_Col1 - hv_Col2).TupleFabs();
    hv_qr_yy = (hv_Row1 - hv_Row2).TupleFabs();



    t2 = clock();
    //std::cout << "qr usetime:" << to_string(t2 - t1) << ",qr_code_str:" << qr_code_str << ",hv_qr_xx:" << to_string(hv_qr_xx.D())<< ",hv_qr_yy:"<< to_string(hv_qr_yy.D())<< endl;
    SPDLOG_INFO("HALCON qr usetime:{:d},qr_code_str:{},qr_xx:{:.2f},qr_yy:{:.2f}", t2 - t1, qr_code_str, hv_qr_xx.D(), hv_qr_yy.D());
}

//启动模板比对的线程，每个模板一个线程
void hconAction::ImageThreadStart(unsigned char cameraid)
{
    std::clock_t t1, t2;
    t1 = clock();

    int thread_i = 0;

    if (cameraid == 2) {    //相机2
        //单边检测 单独开线程
        Halcon_thread.push_back(std::thread(detect_deviation, ho_ImageChangeResult));
        
        
        for (thread_i = 0; thread_i < Halcon_moban_all.size(); thread_i++) {
            Halcon_thread.push_back(std::thread(ImageThreadRun_camera2, thread_i, calculate, Halcon_moban_all[thread_i], ho_ImageChangeResult, now_dealImg_num));
            
            //二维码检测
            if(Halcon_moban_all[thread_i].now_num == 11){
                Halcon_thread.push_back(std::thread(qr_detect, ho_ImageChangeResult, Halcon_moban_all[thread_i].mid_result.ho_ModelRegion));
            }
        }

        //thread_i = 1;
        //Halcon_thread.push_back(std::thread(ImageThreadRun_camera2, thread_i, calculate, Halcon_moban_all[thread_i], ho_ImageChangeResult, now_dealImg_num));
    }

    t2 = clock();
    SPDLOG_INFO("HALCON ImageThreadStart usetime:{:d}", t2 - t1);
}

//is_detect = 1 有缺陷  is_detect = 0 无缺陷
void hconAction::ImageThreadJoin(vector<Detect_Result>& Result, int& is_detect)
{
    std::clock_t t1, t2;
    t1 = clock();

    int detect_num = 0;
    is_detect = 0;
    for (vector<std::thread>::iterator it = Halcon_thread.begin(); it != Halcon_thread.end(); ++it) {
        it->join();
    }

    detect_num = Halcon_Result.size();
    if (detect_num > 0) {
        Result.insert(Result.end(), Halcon_Result.begin(), Halcon_Result.end());  //结果合并到最终结果中
        Halcon_Result.clear();
        is_detect = 1;   //有缺陷
    }

    Halcon_thread.clear();
    t2 = clock();

    SPDLOG_INFO("---HALCON ImageThreadJoin finish! is_detect:{:d},detect_num:{:d},usetime:{:d}", is_detect, detect_num, t2 - t1);
}

void hconAction::GetDataOfmoban(int num, int& x1, int& y1, int& x2, int& y2)
{
    int realnum = 0;
    for (realnum = 0; realnum <= Halcon_moban_all.size(); realnum++) {
        if (Halcon_moban_all[realnum].now_num == num)    break;
    }

    x1 = Halcon_moban_all[realnum].x1;
    y1 = Halcon_moban_all[realnum].y1;
    x2 = Halcon_moban_all[realnum].x2;
    y2 = Halcon_moban_all[realnum].y2;
}

int hconAction::GetNumOfmoban()
{
    return Halcon_moban_all.size();
}

void hconAction::SetNowImgNum(int num_now)
{
    now_dealImg_num = num_now;
}

//获取二维码检测结果 result:1 有单边问题   result：0 无单边问题
void hconAction::GetQrInfo(unsigned char cameraid, string& qr_str, float qr_danbian_x, float qr_danbian_y, int& result)
{
    if (cameraid == 2) {
        qr_str = qr_code_str;

        //if (0 != (HTuple(int(hv_qr_xx > 2.5)).TupleOr(int(hv_qr_yy > 2.5))))
        if (0 != (HTuple(int(hv_qr_xx > qr_danbian_x)).TupleOr(int(hv_qr_yy > qr_danbian_y))))
        {
            result = 1; //有二维码单边问题
        }
    }

    qr_code_str.clear();
}

//获取单边检测结果 result：1 有单边问题
void hconAction::GetDanBianInfo(unsigned char cameraid, float danbian_config, int &result)
{
    if (cameraid == 2) {
        if (0 != (int(hv_Deviation > danbian_config))) {
            result = 1;
        }else {
            result = 0;
        }
    }
}

//样品图片前处理
void hconAction::ImgPreDeal(cv::Mat& oriMatImage)
{
    HalconCpp::HObject ho_ImageInput;
    std::clock_t t1, t2, t3;
    t1 = clock();

    ho_ImageInput = MatToHObject(oriMatImage);   //原始图片转换为halcon格式的图片

    t2 = clock();

    ImageChange_result = ImageChange(ho_ImageInput, ho_ImageChangeResult);  //做图像变换

    t3 = clock();

    SPDLOG_INFO("+++HALCON ImgPreDealThread finish! img:{:d},result:{:d},MatToHObject usetime:{:d},ImageChange usetime:{:d}", now_dealImg_num, ImageChange_result, t2 - t1, t3 - t2);
}

void hconAction::ImgPreDealThread(cv::Mat& oriMatImage)
{
    ImageChange_result = 0;
    //ho_ImageChangetemp.Clear();
    auto ImageChange_thread = std::bind(&hconAction::ImgPreDeal, this, oriMatImage);
    this->ImageChange_thread = new std::thread(ImageChange_thread);
}

int hconAction::ImgPreDealJoin()
{
    ImageChange_thread->join();
    return ImageChange_result;
}