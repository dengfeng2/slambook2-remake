#include "DBoW3/DBoW3.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>

using namespace std;

/***************************************************
 * 本节演示了如何根据data/目录下的十张图训练字典
 * ************************************************/

int main(int argc, char *argv[]) {
    // read the image
    cout << "reading images... " << endl;
    vector<cv::Mat> images;
    for (int i = 0; i < 10; i++) {
        string path = "./data/" + to_string(i + 1) + ".png";
        images.push_back(cv::imread(path));
    }
    // detect ORB features
    cout << "detecting ORB features ... " << endl;
    cv::Ptr<cv::Feature2D> detector = cv::ORB::create();
    vector<cv::Mat> descriptors;
    for (const cv::Mat &image: images) {
        vector<cv::KeyPoint> keyPoints;
        cv::Mat descriptor;
        detector->detectAndCompute(image, cv::Mat(), keyPoints, descriptor);
        descriptors.push_back(descriptor);
    }

    // create vocabulary
    cout << "creating vocabulary ... " << endl;
    DBoW3::Vocabulary vocab;
    vocab.create(descriptors);
    cout << "vocabulary info: " << vocab << endl;
    vocab.save("vocabulary.yml.gz");
    cout << "done" << endl;

    return 0;
}
