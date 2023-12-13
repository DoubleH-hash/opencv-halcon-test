//������ģ��,�������ݽṹ����λ��,�������Ż�

#ifndef COMMONCONTENT_H
#define COMMONCONTENT_H

//#include <iosfwd>
#include <string>
#include <set>
#include <sstream>



using namespace std;


enum CommandType
{
    TCP_UNKNOWN_COMMAND = 0,
    TCP_CONFIG_NODE = 1,
    TCP_SUBSCRIBE_NODE = 2,
    TCP_START_NODE = 3,
    TCP_STOP_NODE = 4,
    TCP_SNAP_IMAGE = 5,
    TCP_CONFIG_SET = 6,
};


enum Node_Type
{
    UNKNOW_TYPE  = 0,
    EN_2D_HIKVISION = 1,
    EN_3D_HIKVISION = 2,
    EN_3D_LMI       = 3,
    EN_ALGORITHM_INTERFACE = 4,
	EN_2D_DALSA = 5,
};

//Grabbing Node Info
struct NodeBasicInfo
{
    Node_Type m_eNodeType;
    string m_strStation;
    string m_strFace;

    NodeBasicInfo()
    {
        m_eNodeType = Node_Type::UNKNOW_TYPE;
        m_strStation = "";
        m_strFace = "A";
    }
};

//�洢�ɼ����Ľ����Ϣ
struct Detect_Result
{
    string result_str;
    int class_res;
    int x1;
    int y1;
    int x2;
    int y2;
    Detect_Result() {
        result_str = "";
        class_res = 0;
        x1 = 0;
        x2 = 0;
        y1 = 0;
        y2 = 0;
    }
};

struct PLC_result_info
{
    int iDBBlock;
    int uniFrameNum;
    PLC_result_info(int iDBBlock, int uniFrameNum) :iDBBlock(iDBBlock),uniFrameNum(uniFrameNum){

    }
};

//��ά��ͼƬ��������
struct QR_Set_Position{
    unsigned int x1, y1;   //���Ͻ�����
    unsigned int x2, y2;    //���½�����

    QR_Set_Position(){
        x1 = 390; y1 = 635;
        x2 = 610; y2 = 840;
    }
};

struct QR_Result {
    int result;      // 1:�ö�ά������   2:�ö�ά���ظ�����  3:�ö�ά��������в�����  4:ͼƬ��δ�ҵ���ά��  5:���ݿ�����ʧ��
    int qr_db_result; //0:��ά�뵥��ȱ��  1��ok  2��err
    string qr_code;
    int qr_x1;    //�ҵ��Ķ�ά�����Ͻ�����
    int qr_y1;
    int qr_x2;   //�ҵ��Ķ�ά�����½�����
    int qr_y2;

    QR_Result() {
        result = 0;
        qr_code = "";
        qr_db_result = -1;
        qr_x1 = 0;
        qr_y1 = 0;
        qr_x2 = 0;
        qr_y2 = 0;
    }
};



struct NodeDetailInfo
{
	//string m_strSerialNum;
	string m_strConfigFileName;	//�����ļ����ƣ�����·����
	string m_strServerName;  //�ɼ�����������
	string m_strCaptureMode;	//�ɼ���ģʽ
    int m_iCaptureIndex;
    string m_strCompletionName;

	string m_strImgType;
    string m_strSaveImgDir;

    NodeDetailInfo()
    {
		//m_strSerialNum = "";
		m_strConfigFileName = "";
		m_strServerName = "";
        m_strCaptureMode = "";
        m_iCaptureIndex = 0;
        m_strCompletionName = "";

		m_strImgType = "bmp";
        m_strSaveImgDir = "";

    }
};





#endif