#include "imagemerger.h"

#include <QDebug>

ImageMerger::ImageMerger()
{

}


// 横向拼接
Mat ImageMerger:: getMerageImageHorizontal(Mat &cur_frame, Mat &cur_frame1)
{
    cv::Mat image_one=cur_frame;
    cv::Mat image_two=cur_frame1;
    cv::Mat img_merge(image_one.rows,image_one.cols + image_two.cols+1,image_one.type());
    image_one.colRange(0,image_one.cols).copyTo(img_merge.colRange(0,image_one.cols));
    image_two.colRange(0,image_two.cols).copyTo(img_merge.colRange(image_one.cols+1,img_merge.cols));
    return img_merge;
}
Mat ImageMerger::getMerageImageHorizontal1(Mat &cur_frame, Mat &cur_frame1)
{
    cv::Mat img_merge;
    cv::Size size(cur_frame.cols + cur_frame1.cols, MAX(cur_frame.rows, cur_frame1.rows));
    img_merge.create(size, CV_MAKETYPE(cur_frame.depth(), 3));
    img_merge = cv::Scalar::all(0);
    cv::Mat outImg_left, outImg_right;
    outImg_left = img_merge(cv::Rect(0, 0, cur_frame.cols, cur_frame.rows));
    outImg_right = img_merge(cv::Rect(cur_frame.cols, 0, cur_frame.cols, cur_frame.rows));
    cur_frame.copyTo(outImg_left);
    cur_frame1.copyTo(outImg_right);
    return img_merge;
}


// 纵向拼接
Mat ImageMerger::getMerageImageVertical(Mat &cur_frame, Mat &cur_frame1)
{
    cv::Mat image_one=cur_frame;
    cv::Mat image_two=cur_frame1;
    cv::Mat img_merge(image_one.rows + image_two.rows, image_one.cols, image_one.type());
    qDebug() << img_merge.rows <<img_merge.cols;
    image_one.rowRange(0,image_one.rows).copyTo(img_merge.rowRange(0,image_one.rows));
    image_two.rowRange(0,image_two.rows).copyTo(img_merge.rowRange(image_one.rows, img_merge.rows));
    return img_merge;
}

Mat ImageMerger::getMerageImageVertical1(Mat &cur_frame, Mat &cur_frame1)
{
    cv::Mat img_merge;
    cv::Size size(MAX(cur_frame.cols , cur_frame1.cols), cur_frame.rows + cur_frame1.rows);
    img_merge.create(size, CV_MAKETYPE(cur_frame.depth(), 3));
    img_merge = cv::Scalar::all(0);
    cv::Mat outImg_left, outImg_right;
    outImg_left = img_merge(cv::Rect(0, 0, cur_frame.cols, cur_frame.rows));
    outImg_right = img_merge(cv::Rect(0,  cur_frame.rows, cur_frame1.cols, cur_frame1.rows));
    cur_frame.copyTo(outImg_left);
    cur_frame1.copyTo(outImg_right);
    return img_merge;
}
Mat ImageMerger::getMerageImageBasedOnTemplate1(const Mat &image1, const Mat &image2)
{
    /*读入图像*/
     //Mat image1 = imread("a.png");
     //Mat image2 = imread("b.png");

    /*转灰度图像*/
    Mat image1_gray, image2_gray;
    cvtColor(image1, image1_gray, CV_BGR2GRAY);
    cvtColor(image2, image2_gray, CV_BGR2GRAY);
    qDebug() << image1.rows << image1.cols;
    qDebug() << image2.rows << image2.cols;

    //imshow("ima", image1);
    //imshow("gray", image1_gray);
    /*取图像2的全部行，1到35列作为模板
    这样image1作为原图，temp作为模板图像
    */
    Mat temp = image2_gray(Range(1, 35), Range::all());
    qDebug() << temp.rows << temp.cols;

    /*结果矩阵图像,大小，数据类型*/
    Mat res(image1_gray.rows - temp.rows + 1, image2_gray.cols - temp.cols + 1, CV_32FC1);
    qDebug() << res.rows << res.cols;

    /*模板匹配，采用归一化相关系数匹配*/
    matchTemplate(image1_gray, temp, res, CV_TM_CCOEFF_NORMED);

    /*结果矩阵阈值化处理*/
    threshold(res, res, 0.8, 1, CV_THRESH_TOZERO);
    double minVal, maxVal, thresholdv = 0.8;
    /*查找最大值及位置*/
    Point minLoc, maxLoc;
    minMaxLoc(res, &minVal, &maxVal, &minLoc, &maxLoc);

    cout << maxVal << endl;
    qDebug() << minLoc.x << minLoc.y;
    qDebug() << maxLoc.x << maxLoc.y;
    //maxLoc.y = 291;
    /*图像拼接*/
    Mat temp1, result;
    if (maxVal >= thresholdv)//只有度量值大于阈值才认为是匹配
    {
        //result:拼接后的图像
        //temp1:原图1的非模板部分
        //result = Mat::zeros(cvSize(maxLoc.x + image2.cols, image1.rows), image1.type());
        result = Mat::zeros(cvSize(image1.cols, maxLoc.y + image2.rows), image1.type());
         qDebug() <<"  ---  " << result.cols << result.rows;
        temp1 = image1(Rect(0, 0, image1.cols, maxLoc.y));
        /*将图1的非模板部分和图2拷贝到result*/
        temp1.copyTo(Mat(result, Rect(0, 0, image1.cols, maxLoc.y)));
        image2.copyTo(Mat(result, Rect(0, maxLoc.y - 1, image2.cols, image2.rows)));
    }

    //imshow("name", result);
    return result;
}

//int main()
Mat ImageMerger::getMerageImageBasedOnTemplate0(const Mat &image1, const Mat &image2)
{
    /*读入图像*/
    //Mat image1 = imread("23.png");
    //Mat image2 = imread("34.png");

    /*转灰度图像*/
    Mat image1_gray, image2_gray;
    cvtColor(image1, image1_gray, CV_BGR2GRAY);
    cvtColor(image2, image2_gray, CV_BGR2GRAY);
    qDebug() << image1.rows << image1.cols;
    qDebug() << image2.rows << image2.cols;

    //imshow("ima", image1);
    //imshow("gray", image1_gray);
    /*取图像2的全部行，1到35列作为模板
    这样image1作为原图，temp作为模板图像
    */
    Mat temp = image2_gray(Range::all(), Range(1, 35));
    qDebug() << temp.rows << temp.cols;

    /*结果矩阵图像,大小，数据类型*/
    Mat res(image1_gray.rows - temp.rows + 1, image2_gray.cols - temp.cols + 1, CV_32FC1);
    qDebug() << res.rows << res.cols;
    /*模板匹配，采用归一化相关系数匹配*/
    matchTemplate(image1_gray, temp, res, CV_TM_CCOEFF_NORMED);

    /*结果矩阵阈值化处理*/
    threshold(res, res, 0.8, 1, CV_THRESH_TOZERO);
    double minVal, maxVal, threshold = 0.8;
    /*查找最大值及位置*/
    Point minLoc, maxLoc;
    minMaxLoc(res, &minVal, &maxVal, &minLoc, &maxLoc);
    qDebug() << minLoc.x << minLoc.y;
    qDebug() << maxLoc.x << maxLoc.y;

    //cout << maxVal << endl;
    /*图像拼接*/
    Mat temp1, result;
    if (maxVal >= threshold)//只有度量值大于阈值才认为是匹配
    {
        //result:拼接后的图像
        //temp1:原图1的非模板部分
        result = Mat::zeros(cvSize(maxLoc.x + image2.cols, image1.rows), image1.type());
        qDebug() <<"  ---  " << result.cols << result.rows;
        temp1 = image1(Rect(0, 0, maxLoc.x, image1.rows));
        /*将图1的非模板部分和图2拷贝到result*/
        temp1.copyTo(Mat(result, Rect(0, 0, maxLoc.x, image1.rows)));
        image2.copyTo(Mat(result, Rect(maxLoc.x - 1, 0, image2.cols, image2.rows)));
    }

    //imshow("name", result);
    return result;
}

Mat ImageMerger::getMerageImageBasedOnTemplate(const Mat &mat1, const Mat &mat2)
{
    Mat imgL = mat1;
    Mat imgR = mat2;

    Mat grayL, grayR;
    cvtColor(imgL, grayL, COLOR_BGR2GRAY);
    cvtColor(imgR, grayR, COLOR_BGR2GRAY);

    Rect rectCut = Rect(0, mat1.rows - 10, mat1.cols, 10);
    Rect rectMatched = Rect(0, 0, imgR.cols / 2, imgR.rows);
    Mat imgTemp = grayL(Rect(rectCut));
    Mat imgMatched = grayR(Rect(rectMatched));

    int width = imgMatched.cols - imgTemp.cols + 1;
    int height = imgMatched.rows - imgTemp.rows + 1;
    Mat matchResult(height, width, CV_32FC1);
    matchTemplate(imgMatched, imgTemp, matchResult, TM_CCORR_NORMED);
    normalize(matchResult, matchResult, 0, 1, NORM_MINMAX, -1);  //归一化到0--1范围

    double minValue, maxValue;
    Point minLoc, maxLoc;
    minMaxLoc(matchResult, &minValue, &maxValue, &minLoc, &maxLoc);

    Mat dstImg(imgL.rows, imgR.cols + rectCut.x - maxLoc.x, CV_8UC3, Scalar::all(0));
    Mat roiLeft = dstImg(Rect(0, 0, imgL.cols, imgL.rows));
    imgL.copyTo(roiLeft);

    Mat debugImg = imgR.clone();
    rectangle(debugImg, Rect(maxLoc.x, maxLoc.y, imgTemp.cols, imgTemp.rows), Scalar(0, 255, 0), 2, 8);
    imwrite("match.jpg", debugImg);

    Mat roiMatched = imgR(Rect(maxLoc.x, maxLoc.y - rectCut.y, imgR.cols - maxLoc.x, imgR.rows - 1 - (maxLoc.y - rectCut.y)));
    Mat roiRight = dstImg(Rect(rectCut.x, 0, roiMatched.cols, roiMatched.rows));

    roiMatched.copyTo(roiRight);
    imshow("dst.jpg", dstImg);

    return dstImg;
}
/*
Mat ImageMerger::getMerageImageOverlap(Mat &img1, Mat &img2)
{
    Mat g1(img1, Rect(0, 0, img1.cols, img1.rows));  // init roi
    Mat g2(img2, Rect(0, 0, img2.cols, img2.rows));

    cvtColor(g1, g1, COLOR_BGR2GRAY);
    cvtColor(g2, g2, COLOR_BGR2GRAY);

    vector<cv::KeyPoint> keypoints_roi, keypoints_img;
    Mat descriptor_roi, descriptor_img;
    FlannBasedMatcher matcher;
    vector<cv::DMatch> matches, good_matches;
    cv::Ptr<cv::SIFT> sift = cv::SIFT::create();
    int i, dist = 80;
    sift->detectAndCompute(g1, cv::Mat(), keypoints_roi, descriptor_roi);
    sift->detectAndCompute(g2, cv::Mat(), keypoints_img, descriptor_img);
    matcher.match(descriptor_roi, descriptor_img, matches);  //实现描述符之间的匹配

    double max_dist = 0; double min_dist = 5000;
    //-- Quick calculation of max and min distances between keypoints
    for (int i = 0; i < descriptor_roi.rows; i++)
    {
        double dist = matches[i].distance;
        if (dist < min_dist) min_dist = dist;
        if (dist > max_dist) max_dist = dist;
    }
    // 特征点筛选
    for (i = 0; i < descriptor_roi.rows; i++)
    {
        if (matches[i].distance < 3 * min_dist)
        {
            good_matches.push_back(matches[i]);
        }
    }

    Mat img_matches;
    //绘制匹配
    drawMatches(img1, keypoints_roi, img2, keypoints_img,
                good_matches, img_matches, Scalar::all(-1),
                Scalar::all(-1), vector<char>(),
                DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    imshow("matches", img_matches);

    vector<Point2f> keypoints1, keypoints2;
    for (i = 0; i < good_matches.size(); i++)
    {
        keypoints1.push_back(keypoints_img[good_matches[i].trainIdx].pt);
        keypoints2.push_back(keypoints_roi[good_matches[i].queryIdx].pt);
    }
    //计算单应矩阵(仿射变换矩阵)
    Mat H = findHomography(keypoints1, keypoints2, RANSAC);
    Mat H2 = findHomography(keypoints2, keypoints1, RANSAC);


    Mat stitchedImage;  //定义仿射变换后的图像(也是拼接结果图像)
    Mat stitchedImage2;  //定义仿射变换后的图像(也是拼接结果图像)
    int mRows = img2.rows;
    if (img1.rows > img2.rows)
    {
        mRows = img1.rows;
    }

    int count = 0;
    for (int i = 0; i < keypoints2.size(); i++)
    {
        if (keypoints2[i].x >= img2.cols / 2)
            count++;
    }
    //判断匹配点位置来决定图片是左还是右
    if (count / float(keypoints2.size()) >= 0.5)  //待拼接img2图像在右边
    {
        cout << "img1 should be left" << endl;
        vector<Point2f>corners(4);
        vector<Point2f>corners2(4);
        corners[0] = Point(0, 0);
        corners[1] = Point(0, img2.rows);
        corners[2] = Point(img2.cols, img2.rows);
        corners[3] = Point(img2.cols, 0);
        stitchedImage = Mat::zeros(img2.cols + img1.cols, mRows, CV_8UC3);
        warpPerspective(img2, stitchedImage, H, Size(img2.cols + img1.cols, mRows));

        perspectiveTransform(corners, corners2, H);
        cout << corners2[0].x << ", " << corners2[0].y << endl;
        cout << corners2[1].x << ", " << corners2[1].y << endl;
        imshow("temp", stitchedImage);
        //imwrite("temp.jpg", stitchedImage);

        Mat half(stitchedImage, Rect(0, 0, img1.cols, img1.rows));
        img1.copyTo(half);
        imshow("result", stitchedImage);
    }
    else  //待拼接图像img2在左边
    {
        cout << "img2 should be left" << endl;
        stitchedImage = Mat::zeros(img2.cols + img1.cols, mRows, CV_8UC3);
        warpPerspective(img1, stitchedImage, H2, Size(img1.cols + img2.cols, mRows));
        imshow("temp", stitchedImage);

        //计算仿射变换后的四个端点
        vector<Point2f>corners(4);
        vector<Point2f>corners2(4);
        corners[0] = Point(0, 0);
        corners[1] = Point(0, img1.rows);
        corners[2] = Point(img1.cols, img1.rows);
        corners[3] = Point(img1.cols, 0);

        perspectiveTransform(corners, corners2, H2);  //仿射变换对应端点
        cout << corners2[0].x << ", " << corners2[0].y << endl;
        cout << corners2[1].x << ", " << corners2[1].y << endl;

        Mat half(stitchedImage, Rect(0, 0, img2.cols, img2.rows));
        img2.copyTo(half);
        imshow("result", stitchedImage);

    }
    imwrite("result.bmp", stitchedImage);
    return stitchedImage;
}
*/
