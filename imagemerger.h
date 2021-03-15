#ifndef IMAGEMERGER_H
#define IMAGEMERGER_H
#include<opencv2/opencv.hpp>
using namespace cv;
using namespace std;

class ImageMerger
{
public:
    ImageMerger();


    // 硬拼接
    // 横向拼接
    static Mat getMerageImageHorizontal(Mat &cur_frame, Mat &cur_frame1);
    static Mat getMerageImageHorizontal1(Mat &cur_frame, Mat &cur_frame1);
    // 纵向拼接
    static Mat getMerageImageVertical(Mat &cur_frame, Mat &cur_frame1);
    static Mat getMerageImageVertical1(Mat &cur_frame, Mat &cur_frame1);


    // 基于模板匹配拼接
    static Mat getMerageImageBasedOnTemplate(const Mat &mat1, const Mat &mat2);
    static Mat getMerageImageBasedOnTemplate0(const Mat &image1, const Mat &image2);
    static Mat getMerageImageBasedOnTemplate1(const Mat &image1, const Mat &image2);
    // 基于特征匹配拼接
    //static Mat getMerageImageOverlap(Mat &img1, Mat &img2);

};

#endif // IMAGEMERGER_H
