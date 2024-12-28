#include "DBoW3/DBoW3.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>

using namespace std;

/***************************************************
 * 本节演示了如何根据前面训练的字典计算相似性评分
 * ************************************************/
int main(int argc, char *argv[]) {
    // read the images and database
    cout << "reading database" << endl;
    DBoW3::Vocabulary vocab("./vocab_larger.yml.gz");
    // DBoW3::Vocabulary vocab("./vocab_larger.yml.gz");  // use large vocab if you want:
    if (vocab.empty()) {
        cerr << "Vocabulary does not exist." << endl;
        return 1;
    }
    cout << "reading images... " << endl;
    vector<cv::Mat> images;
    for (int i = 0; i < 10; i++) {
        string path = "./data/" + to_string(i + 1) + ".png";
        images.push_back(cv::imread(path));
    }

    // NOTE: in this case we are comparing images with a vocabulary generated by themselves, this may lead to overfit.
    // detect ORB features
    cout << "detecting ORB features ... " << endl;
    cv::Ptr<cv::Feature2D> detector = cv::ORB::create();
    vector<cv::Mat> descriptors;
    for (const cv::Mat &image: images) {
        vector<cv::KeyPoint> keypoints;
        cv::Mat descriptor;
        detector->detectAndCompute(image, cv::Mat(), keypoints, descriptor);
        descriptors.push_back(descriptor);
    }

    // we can compare the images directly or we can compare one image to a database
    // images :
    cout << "comparing images with images " << endl;
    for (int i = 0; i < images.size(); i++) {
        DBoW3::BowVector v1;
        vocab.transform(descriptors[i], v1);
        for (int j = i; j < images.size(); j++) {
            DBoW3::BowVector v2;
            vocab.transform(descriptors[j], v2);
            double score = vocab.score(v1, v2);
            cout << "image " << i << " vs image " << j << " : " << score << endl;
        }
        cout << endl;
    }

    // or with database
    cout << "comparing images with database " << endl;
    DBoW3::Database db(vocab, false, 0);
    for (const auto & descriptor : descriptors) {
        db.add(descriptor);
    }
    cout << "database info: " << db << endl;
    for (int i = 0; i < descriptors.size(); i++) {
        DBoW3::QueryResults ret;
        db.query(descriptors[i], ret, 4); // max result=4
        cout << "searching for image " << i << " returns " << ret << endl << endl;
    }
    cout << "done." << endl;
}