#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>

using namespace std;

int main(int argc, const char *argv[]) {
    cv::Mat image = cv::imread(argv[1]);
    if (image.empty()) {
        cerr << "文件" << argv[1] << "不存在." << endl;
        return -1;
    }
    cout << "图像宽为" << image.cols << ",高为" << image.rows << ",通道数为" << image.channels() << endl;
    cv::imshow("image", image); // 用cv::imshow显示图像
    cv::waitKey(0); // 暂停程序,等待一个按键输入

    // 判断image的类型
    if (image.type() != CV_8UC1 && image.type() != CV_8UC3) {
        // 图像类型不符合要求
        cout << "请输入一张彩色图或灰度图." << endl;
        return 0;
    }

    // 遍历图像, 请注意以下遍历方式亦可使用于随机像素访问
    // 使用 std::chrono 来给算法计时

    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    for (auto y = 0; y < image.rows; y++) {
        auto *row_ptr = image.ptr<unsigned char>(y); // row_ptr是第y行的头指针
        for (size_t x = 0; x < image.cols; x++) {
            unsigned char *data_ptr = &row_ptr[x * image.channels()]; // data_ptr 指向待访问的像素数据
            for (int c = 0; c != image.channels(); c++) {
                unsigned char data = data_ptr[c]; // data为I(x,y)第c个通道的值
            }
        }
    }
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double> >(t2 - t1);
    cout << "遍历图像用时：" << time_used.count() << " 秒。" << endl;

    // 关于 cv::Mat 的拷贝
    // 直接赋值并不会拷贝数据
    cv::Mat image_another = image;
    // 修改 image_another 会导致 image 发生变化
    image_another(cv::Rect(0, 0, 100, 100)).setTo(0); // 将左上角100*100的块置零
    cv::imshow("image", image);
    cv::waitKey(0);

    // 使用clone函数来拷贝数据
    cv::Mat image_clone = image.clone();
    image_clone(cv::Rect(0, 0, 100, 100)).setTo(255);
    cv::imshow("image", image);
    cv::imshow("image_clone", image_clone);
    cv::waitKey(0);

    // 对于图像还有很多基本的操作,如剪切,旋转,缩放等,限于篇幅就不一一介绍了,请参看OpenCV官方文档查询每个函数的调用方法.
    cv::destroyAllWindows();
    return 0;
}
