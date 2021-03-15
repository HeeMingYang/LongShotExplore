
#include "imagemerger.h"

#include <QCoreApplication>
#include <QString>
#include <QDebug>
#include <QDir>
#include <QTime>
/*
int main(int argc, char *argv[])
{
    QCoreApplication a(argc, argv);

    return a.exec();
}
*/

#include<opencv2/opencv.hpp>
//#include<opencv2/highgui/highgui.hpp>"
//#include<opencv2/xfeatures2d/nonfree.hpp>
//#include<opencv2/legacy/legacy.hpp>
#include<iostream>
using namespace cv;
using namespace std;
/*
int surfImage()
{
     Mat image01 = imread("2.jpg", 1);    //右图
     Mat image02 = imread("1.jpg", 1);    //左图
     namedWindow("p2", 0);
     namedWindow("p1", 0);
     imshow("p2", image01);
     imshow("p1", image02);

     //灰度图转换
     Mat image1, image2;
     cvtColor(image01, image1, CV_RGB2GRAY);
     cvtColor(image02, image2, CV_RGB2GRAY);


     //提取特征点
     SurfFeatureDetector surfDetector(800);  // 海塞矩阵阈值，在这里调整精度，值越大点越少，越精准
     vector<KeyPoint> keyPoint1, keyPoint2;
     surfDetector.detect(image1, keyPoint1);
     surfDetector.detect(image2, keyPoint2);

     //特征点描述，为下边的特征点匹配做准备
     SurfDescriptorExtractor SurfDescriptor;
     Mat imageDesc1, imageDesc2;
     SurfDescriptor.compute(image1, keyPoint1, imageDesc1);
     SurfDescriptor.compute(image2, keyPoint2, imageDesc2);

     //获得匹配特征点，并提取最优配对
     FlannBasedMatcher matcher;
     vector<DMatch> matchePoints;

     matcher.match(imageDesc1, imageDesc2, matchePoints, Mat());
     cout << "total match points: " << matchePoints.size() << endl;


     Mat img_match;
     drawMatches(image01, keyPoint1, image02, keyPoint2, matchePoints, img_match);
     namedWindow("match", 0);
     imshow("match",img_match);
     imwrite("match.jpg", img_match);

     waitKey();
}
*/
// 比较图片m1的r1行像素和m2的r2行像素是否相等
bool CompareRowsData(const Mat &m1, const int r1, const Mat &m2, const int r2)
{
    if(m1.cols != m2.cols)
        return false;

    for(int i = 0; i < m1.cols; ++i) {
        if( m1.at<Vec3b>(r1, i)[0] != m2.at<Vec3b>(r2, i)[0]||
                m1.at<Vec3b>(r1, i)[1] != m2.at<Vec3b>(r2, i)[1]||
                m1.at<Vec3b>(r1, i)[2] != m2.at<Vec3b>(r2, i)[2]
                ) {
            //qDebug() << r1 << r2 <<i;
            return false;
        }
    }
    return true;
}
// 获取两张图片变化区域
void getChangeArea(QString img1Path, QString img2Path, Mat &c1, Mat &c2)
{
    //从两张图片中截取变化的区域
    Mat img1=imread(img1Path.toLocal8Bit().data());
    Mat img2=imread(img2Path.toLocal8Bit().data());
    Mat dst;//存储结果
    cout<<"img1  "<<int(img1.at<Vec3b>(10,10)[0])<<endl;//img1在坐标（10,10）的蓝色通道的值，强制转成int
    cout<<"img2  "<<int(img2.at<Vec3b>(10,10)[0])<<endl;

    int minI = img1.rows;
    int minJ = img1.cols;
    int maxI = 0;
    int maxJ = 0;
    // 计算变化部分
    for(int i = 0; i < img1.rows; ++i) {
        for(int j = 0; j < img1.cols; ++j) {
            if(img1.at<Vec3b>(i, j)[0] != img2.at<Vec3b>(i, j)[0] || img1.at<Vec3b>(i, j)[1] != img2.at<Vec3b>(i, j)[1] || img1.at<Vec3b>(i, j)[2] != img2.at<Vec3b>(i, j)[2]) {
                //cout << "rows:" << i << "cols:" << j << endl; //<< img1.at<Vec3b>(j, i) << img2.at<Vec3b>(j, i) << endl;
                if( i < minI) minI = i;
                if( j < minJ) minJ = j;
                if( i > maxI) maxI = i;
                if( j > maxJ) maxJ = j;

            }
        }
    }
    cout << "("<< minJ  << " "<< minI << ")"<< "(" << maxJ - minJ << " " << maxI - minI<< ")" << endl;
    // 截取变化部分
    c1 =  Mat(img1, Rect(minJ, minI, maxJ - minJ, maxI - minI));
    c2 =  Mat(img2, Rect(minJ, minI, maxJ - minJ, maxI - minI));
}
// 获取两张图片变化区域
Rect getChangeArea(Mat &img1, Mat &img2)
{
    cout<<"img1  "<<int(img1.at<Vec3b>(10,10)[0])<<endl;//img1在坐标（10,10）的蓝色通道的值，强制转成int
    cout<<"img2  "<<int(img2.at<Vec3b>(10,10)[0])<<endl;

    int minI = img1.rows;
    int minJ = img1.cols;
    int maxI = 0;
    int maxJ = 0;
    // 计算变化部分
    for(int i = 0; i < img1.rows; ++i) {
        for(int j = 0; j < img1.cols; ++j) {
            if(img1.at<Vec3b>(i, j)[0] != img2.at<Vec3b>(i, j)[0] || img1.at<Vec3b>(i, j)[1] != img2.at<Vec3b>(i, j)[1] || img1.at<Vec3b>(i, j)[2] != img2.at<Vec3b>(i, j)[2]) {
                //cout << "rows:" << i << "cols:" << j << endl; //<< img1.at<Vec3b>(j, i) << img2.at<Vec3b>(j, i) << endl;
                if( i < minI) minI = i;
                if( j < minJ) minJ = j;
                if( i > maxI) maxI = i;
                if( j > maxJ) maxJ = j;

            }
        }
    }

    return  Rect(minJ, minI, maxJ - minJ, maxI - minI);
}
// openCV 融合拼接算法
bool mergeImgOpenCV(Mat &pano, Mat &img1, Mat &img2)
{
    vector<Mat> imgs;
    imgs.push_back(img1);
    imgs.push_back(img2);

    Ptr<Stitcher> stitcher = Stitcher::create(Stitcher::SCANS, true);
    //Ptr<Stitcher> stitcher = Stitcher::create(Stitcher::PANORAMA, true);
    Stitcher::Status status = stitcher->stitch(imgs, pano);
    if(status == Stitcher::OK) {
        return  true;
    }
    return false;
}

// openCV 融合拼接算法
bool mergeImgOpenCV(Mat &pano, vector<Mat> &imgs)
{
    Ptr<Stitcher> stitcher = Stitcher::create(Stitcher::SCANS, true);
    //Ptr<Stitcher> stitcher = Stitcher::create(Stitcher::PANORAMA, true);
    Stitcher::Status status = stitcher->stitch(imgs, pano);
    if(status == Stitcher::OK) {
        return  true;
    }
    return false;
}

bool mergeImg(Mat &pano, Mat &img1, Mat &img2)
{
    //变化部分重复计算
    int i = 0;
    int j = 0;
    int k = 0;

    int start = 0;
    bool find = false;
    for(i = 0; i < img1.rows; ++i) {
        find = true;
        for(j = 0; j < img1.cols; ++j) {
            if(img1.at<Vec3b>(img1.rows, j)[0] != img2.at<Vec3b>(i, j)[0] ||
                    img1.at<Vec3b>(img1.rows, j)[1] != img2.at<Vec3b>(i, j)[1] ||
                    img1.at<Vec3b>(img1.rows, j)[2] != img2.at<Vec3b>(i, j)[2]) {
                find = false;
                continue;
            }
        }
        if(find)
            goto S;
    }
S:
    cout <<"s " << i << endl;


    pano = Mat(img2, Rect(0, 0, img2.cols, img2.rows));
    return true;
}

Mat getMatByRect(QString str, Rect &rect)
{
    Mat img = imread(str.toLocal8Bit().data());
    return Mat(img, rect);
}


cv::Mat get_merage_image(cv::Mat cur_frame, cv::Mat cur_frame1)
{
    cv::Mat image_one=cur_frame;
    cv::Mat image_two=cur_frame1;
    //创建连接后存入的图像，两幅图像按左右排列，所以列数+1
    cv::Mat img_merge(image_one.rows,image_one.cols+
                      image_two.cols+1,image_one.type());
    //图像拷贝，不能用Mat中的clone和copyTo函数，单幅图像拷贝可用，clone和copyTo不仅拷贝图像数据，还拷贝一///些其他的信息
    //而现在是将两幅图像的数据拷贝到一副图像中，只拷贝图像数据
    //因此用colRange来访问图像的列数据colRange第一参数是起始列，是从0开始索引，而第二个参数是结束列，
    //从1开始索引，与我们以前使用的不同，因此，参数分别为0和image_one.cols
    image_one.colRange(0,image_one.cols).
            copyTo(img_merge.colRange(0,image_one.cols));
    //第二幅图像拷贝,中间的一行作为两幅图像的分割线
    image_two.colRange(0,image_two.cols).copyTo(
                img_merge.colRange(image_one.cols+1,img_merge.cols));
    return img_merge;
}

cv::Mat get_merage_image2(cv::Mat cur_frame, cv::Mat cur_frame1)
{
    cv::Mat img_merge;
    cv::Size size(cur_frame.cols + cur_frame1.cols, MAX(cur_frame.rows, cur_frame1.rows));
    img_merge.create(size, CV_MAKETYPE(cur_frame.depth(), 3));
    img_merge = cv::Scalar::all(0);
    cv::Mat outImg_left, outImg_right;
    //2.在新建合并图像中设置感兴趣区域
    outImg_left = img_merge(cv::Rect(0, 0, cur_frame.cols, cur_frame.rows));
    outImg_right = img_merge(cv::Rect(cur_frame.cols, 0, cur_frame.cols, cur_frame.rows));
    //3.将待拷贝图像拷贝到感性趣区域中
    cur_frame.copyTo(outImg_left);
    cur_frame1.copyTo(outImg_right);
    return img_merge;
}


cv::Mat getMerageImage(cv::Mat cur_frame, cv::Mat cur_frame1)
{
    cv::Mat image_one=cur_frame;
    cv::Mat image_two=cur_frame1;
    //创建连接后存入的图像，两幅图像按左右排列，所以列数+1
    cv::Mat img_merge(image_one.rows + image_two.rows, image_one.cols, image_one.type());
    qDebug() << img_merge.rows <<img_merge.cols;
    //图像拷贝，不能用Mat中的clone和copyTo函数，单幅图像拷贝可用，clone和copyTo不仅拷贝图像数据，还拷贝一///些其他的信息
    //而现在是将两幅图像的数据拷贝到一副图像中，只拷贝图像数据
    //因此用colRange来访问图像的列数据colRange第一参数是起始列，是从0开始索引，而第二个参数是结束列，
    //从1开始索引，与我们以前使用的不同，因此，参数分别为0和image_one.cols
    image_one.rowRange(0,image_one.rows).copyTo(img_merge.rowRange(0,image_one.rows));
    //第二幅图像拷贝,中间的一行作为两幅图像的分割线
    image_two.rowRange(0,image_two.rows).copyTo(img_merge.rowRange(image_one.rows, img_merge.rows));
    return img_merge;
}

cv::Mat getMerageImage2(cv::Mat cur_frame, cv::Mat cur_frame1)
{
    cv::Mat img_merge;
    cv::Size size(MAX(cur_frame.cols , cur_frame1.cols), cur_frame.rows + cur_frame1.rows);
    img_merge.create(size, CV_MAKETYPE(cur_frame.depth(), 3));
    img_merge = cv::Scalar::all(0);
    cv::Mat outImg_left, outImg_right;
    //2.在新建合并图像中设置感兴趣区域
    outImg_left = img_merge(cv::Rect(0, 0, cur_frame.cols, cur_frame.rows));
    outImg_right = img_merge(cv::Rect(0,  cur_frame.rows, cur_frame1.cols, cur_frame1.rows));
    //3.将待拷贝图像拷贝到感性趣区域中
    cur_frame.copyTo(outImg_left);
    cur_frame1.copyTo(outImg_right);
    return img_merge;
}

bool isBestValue(const Mat &c1, const int r1, const Mat &c2, const int r2)
{
    for (int i = r1 - 1; i > r1 - 7; --i) {
        for (int j = r2 + 1; j < r2 + 7; ++j) {
            if(!CompareRowsData(c1, i, c2, j)) {
                return false;
            }
        }
    }
    return true;
}
void getPicture(const Mat &c1, const Mat &c2, Rect &r1, Rect &r2)
{
    if(c1.cols != c2.cols)
        return;
    for (int i = c1.rows - 1; i >= 0; --i) {
        for (int j = 0; j < c2.rows; ++j) {
            if(CompareRowsData(c1, i, c2, j)) {

                qDebug() << i << j << "sssssssss";
                r1 = Rect(0, 0, c1.cols, i);
                r2 = Rect(0, j, c2.cols, c2.rows - j);
                //if(isBestValue(c1, i, c2, j)) {
                    return;
                //}
            }
        }
    }
}
const int COM_ROW = 5;


// 提取两张图片的变化区域
// 拼接变化区域
void test1(QString img1, QString img2)
{
    Mat c1;
    Mat c2;
    getChangeArea(img1, img2, c1, c2);
    imshow("c1",c1);
    imshow("c2",c2);

    Mat pano;
    if(mergeImgOpenCV(pano, c1, c2)){
        imshow("pano", pano);
    }
}

void test2(QString img1, QString img2)
{
    Mat c1;
    Mat c2;
    getChangeArea(img1, img2, c1, c2);
    imshow("c1",c1);
    imshow("c2",c2);

    Mat pano;
    if(mergeImg(pano, c1, c2)){
        imshow("pano", pano);
    }
}
void getLongPictureFormDir(const QString str)
{
    QDir dir(str);
    if(!dir.exists())
        return;
    // 过滤出目录下的png格式图片
    QString filtername = "*.png";
    QStringList filter;
    filter << filtername;
    dir.setNameFilters(filter);

    QStringList temp_file = dir.entryList();
    qDebug() << temp_file;
    // 图片数小于2不能做拼图
    if(temp_file.size() < 2)
        return;
    // 根据前两张图片，识别出图片滑动区域
    vector<Mat> imgs;
    Mat img1 = imread((str + "/" + temp_file[0]).toLocal8Bit().data());
    Mat img2 = imread((str + "/" + temp_file[1]).toLocal8Bit().data());
    Rect rect = getChangeArea(img1, img2);

    // 从所有图片中，裁剪出滑动区域。并有序存入imgs中
    foreach(QString s , temp_file) {
        QString temp = str + "/" + s;
        imgs.push_back(getMatByRect(temp, rect));
    }
    // 采用基于模板匹配的方式依次拼接图象
    Mat pano;
    pano = imgs[0];
    for(int i = 1; i < imgs.size(); ++i) {
        pano = ImageMerger::getMerageImageBasedOnTemplate1(pano, imgs[i]);
    }
    imshow("pano", pano);
    // 保存拼接好的图片
    imwrite("pano.png", pano);
}



void test4(QString img1, QString img2)
{
    Mat c1;
    Mat c2;
    getChangeArea(img1, img2, c1, c2);

    imshow("c1",c1);
    imshow("c2",c2);
    imwrite("c11.png", c1);
    imwrite("c21.png", c2);
    Mat merger = ImageMerger::getMerageImageBasedOnTemplate1(c1, c2);
    imwrite("merger.png", merger);
    imshow("merger", merger);
    return;

    Rect r1;
    Rect r2;

    getPicture(c1, c2, r1, r2);
    imshow("c11", Mat(c1, r1));
    imshow("c21", Mat(c2, r2));


    Mat s = getMerageImage2(Mat(c1, r1), Mat(c2, r2));
    qDebug() << s.cols << s.rows;
    imshow("merger", s);
    imwrite((img1 + img2 + ".png").toLocal8Bit().data(), s);
}

void test5(QString img1, QString img2)
{
    Mat c1 = imread(img1.toLocal8Bit().data());
    Mat c2 = imread(img2.toLocal8Bit().data());

    imshow("c11", Mat(c1, Rect(0, 0, 1096, 621)));
    imshow("c21", Mat(c2, Rect(0, 291, 1096, 632 - 291 )));


    // imshow("merger", getMerageImage(Mat(c1, Rect(0, 0, 1096, 621)), Mat(c2, Rect(0, 291, 1096, 632 - 291 ))));
    Mat s = getMerageImage2(Mat(c1, Rect(0, 0, 1096, 621)), Mat(c2, Rect(0, 291, 1096, 632 - 291 )));
    qDebug() << s.cols << s.rows;
    imshow("merger", s);
    //for(int j = 0; j < c1.rows; ++j) {
    for(int i = 0; i < c2.rows; ++i) {
        if(CompareRowsData(c1, 621, c2, i)){
            qDebug() << 621 << i;
        }
    }
    //}
}
// 拼接测试
void test6()
{
    Mat img = imread("s");
    Mat img1 = imread("r");
    // 横向拼接
    //Mat imgout = get_merage_image(img, img1);
    // 纵向拼接
    //Mat imgout = getMerageImage(img, img1);
    //Mat imgout = ImageMerger::getMerageImageBasedOnTemplate0(img, img1);
    Mat imgout = ImageMerger::getMerageImageBasedOnTemplate1(img, img1);
    imshow("merge", imgout);
}
int main()
{
    QTime time;
    time.start();
    //getLongPictureFormDir("testImg");
    //test1("a.png", "b.png");
    //test2("a.png", "b.png");
    //test3();
    test4("./a1.png", "./a2.png");
    //test4("b1.png", "b2.png");
    //test6();
    //qDebug()<<"time：" << time.elapsed() << "ms";
    //test5("c1.png", "c2.png");
    //test6();
    waitKey(0);
    return 0;
}


