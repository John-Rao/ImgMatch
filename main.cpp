#include <QCoreApplication>
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/legacy/legacy.hpp" // 暴力匹配的头文件
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include <QDebug>

using namespace cv;
using namespace std;

int main(int argc, char *argv[])
{
    QCoreApplication a(argc, argv);

    //读取待拼接将图像
    Mat img_1=imread("e1.jpg");
    Mat img_2=imread("e2.jpg");
    if(!img_1.data||!img_2.data){
        qDebug()<<"读取图像失败!";
        return -1;
    }
    //侦测特征点surf
    vector<KeyPoint> keypoint_1,keypoint_2;
    Mat descriptors_1,descriptors_2;
    /*/
    SURF surf;
    surf(img_1,Mat(),keypoint_1,descriptors_1);
    surf(img_2,Mat(),keypoint_2,descriptors_2);//*/
    //-----------------ORB featrue Point----------------
    ORB orb;   // float Feature, can not use FlannBase Match.
    orb(img_1, Mat(), keypoint_1, descriptors_1);
    orb(img_2, Mat(), keypoint_2, descriptors_2);
    //*/
    //特征点匹配
    //FlannBasedMatcher matcher;
    BruteForceMatcher<HammingLUT> matcher;// orb 等float型的
    vector<DMatch> matches;
    matcher.match(descriptors_1,descriptors_2,matches);
    //寻找匹配点对距离
    double max_dist=0;
    double min_dist=100;
    for(int i=0;i<descriptors_1.rows;i++){
        double dist=matches[i].distance;
        if(dist<min_dist) min_dist=dist;
        if(dist>max_dist) max_dist=dist;
    }
    qDebug()<<"-- Max distance:"<<max_dist;
    qDebug()<<"-- Min distance:"<<min_dist;
    //寻找较好的匹配点对(0.6*max)
    vector<DMatch> good_matches;
    for(int i=0;i<descriptors_1.rows;i++){
        if(matches[i].distance<0.6*max_dist){
            good_matches.push_back(matches[i]);
        }
    }
    //计算F
    int ptCount = (int)matches.size();
    Mat p1(ptCount, 2, CV_32F);
    Mat p2(ptCount, 2, CV_32F);

    //存储原始信息
    vector<KeyPoint> mateKeypoint_1;
    vector<KeyPoint> mateKeypoint_2;

    Point2f pt;
    for(int i=0;i<ptCount;i++){
        mateKeypoint_1.push_back(keypoint_1[matches[i].queryIdx]);
        pt = mateKeypoint_1.at(i).pt;
        p1.at<float>(i, 0) = pt.x;
        p1.at<float>(i, 1) = pt.y;

        mateKeypoint_2.push_back(keypoint_2[matches[i].trainIdx]);
        pt = mateKeypoint_2.at(i).pt;
        p2.at<float>(i, 0) = pt.x;
        p2.at<float>(i, 1) = pt.y;
    }
    //用RanSAC方法计算基本矩阵F
    Mat m_Fundamental;
    vector<uchar> m_RANSACStatus;
    Mat H;

    m_Fundamental=findFundamentalMat(p1,p2,m_RANSACStatus,FM_RANSAC,3.);
    // 计算野点个数
    int OutlinerCount = 0;
    for (int i = 0; i<ptCount; i++)
    {
        if (m_RANSACStatus[i] == 0) // 状态为0表示野点
        {
            OutlinerCount++;
        }
    }
    // 计算内点,移除噪声
    vector<DMatch> m_InlierMatches;

    int InlinerCount = ptCount - OutlinerCount;
    int removePoint=0;
    m_InlierMatches.resize(InlinerCount);
    InlinerCount = 0;
    for (int i = 0; i<ptCount; i++){
        if (m_RANSACStatus[i] != 0){
            m_InlierMatches[InlinerCount].queryIdx = InlinerCount;
            m_InlierMatches[InlinerCount].trainIdx = InlinerCount;
            InlinerCount++;
        }else{
            mateKeypoint_1.erase(mateKeypoint_1.begin()+i-removePoint);
            mateKeypoint_2.erase(mateKeypoint_2.begin()+i-removePoint);
            removePoint++;
        }
    }
    //获取Mat格式的强匹配点

    Mat rotate_p1(InlinerCount, 2, CV_32F);
    Mat rotate_p2(InlinerCount, 2, CV_32F);
    for(int i=0;i<InlinerCount;i++){
        rotate_p1.at<float>(i,0)=mateKeypoint_1.at(i).pt.x;
        rotate_p1.at<float>(i,1)=mateKeypoint_1.at(i).pt.y;
        rotate_p2.at<float>(i,0)=mateKeypoint_2.at(i).pt.x;
        rotate_p2.at<float>(i,1)=mateKeypoint_2.at(i).pt.y;
    }

    //计算
    H = findHomography( rotate_p2, rotate_p1, CV_RANSAC );

    //绘制匹配点
    qDebug()<<mateKeypoint_1.size()<<"----"<<InlinerCount;
    Mat OutImage;
    drawMatches(img_1, mateKeypoint_1, img_2, mateKeypoint_2, m_InlierMatches, OutImage);
    imshow("outImage",OutImage);

    //绘制透视变换

    vector<Point2f> obj_corners(4);
    obj_corners[0] = Point(0,0);
    obj_corners[1] = Point( img_1.cols, 0 );
    obj_corners[2] = Point( img_1.cols, img_1.rows );
    obj_corners[3] = Point( 0, img_1.rows );
    vector<Point2f> scene_corners(4);
    perspectiveTransform( obj_corners, scene_corners, H);

    Point2f offset( (float)img_1.cols, 0);
    line( OutImage, scene_corners[0] + offset, scene_corners[1] + offset, Scalar(0, 255, 0), 4 );
    line( OutImage, scene_corners[1] + offset, scene_corners[2] + offset, Scalar( 0, 255, 0), 4 );
    line( OutImage, scene_corners[2] + offset, scene_corners[3] + offset, Scalar( 0, 255, 0), 4 );
    line( OutImage, scene_corners[3] + offset, scene_corners[0] + offset, Scalar( 0, 255, 0), 4 );
    // 显示计算F过后的内点匹配

//    imshow("Match2", OutImage);
    //图像对准
    Mat result;
    Mat result_back;
    warpPerspective(img_2,result,H,Size(2*img_2.cols,img_2.rows));
    result.copyTo(result_back);
    Mat half(result,Rect(0,0,img_2.cols,img_2.rows));
    img_1.copyTo(half);
    imshow("result",result);
    //图像融合
    //渐入渐出融合
    Mat result_linerblend = result.clone();
    double dblend = 0.0;
    int ioffset =img_2.cols-100;
    for (int i = 0;i<100;i++){
        result_linerblend.col(ioffset+i) = result.col(ioffset+i)*(1-dblend) + result_back.col(ioffset+i)*dblend;
        dblend = dblend +0.01;
    }
    imshow("result_linerblend",result_linerblend);

    waitKey(0);
    return a.exec();
}









