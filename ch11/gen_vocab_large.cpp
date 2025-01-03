#include "DBoW3/DBoW3.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>

using namespace std;

int main(int argc, char *argv[]) {
    string dataset_dir = argv[1];
    ifstream fin(dataset_dir + "/associate.txt");
    if (!fin) {
        cout << "please generate the associate file called associate.txt!" << endl;
        return 1;
    }

    vector<string> rgb_files, depth_files;
    vector<double> rgb_times, depth_times;
    while (!fin.eof()) {
        double rgb_time, depth_time;
        string rgb_file, depth_file;
        fin >> rgb_time >> rgb_file >> depth_time >> depth_file;
        rgb_times.push_back(rgb_time);
        depth_times.push_back(depth_time);
        rgb_files.push_back(dataset_dir + "/" + rgb_file);
        depth_files.push_back(dataset_dir + "/" + depth_file);

        if (!fin.good())
            break;
    }
    fin.close();

    cout << "generating features ... " << endl;
    vector<cv::Mat> descriptors;
    cv::Ptr<cv::Feature2D> detector = cv::ORB::create();
    int index = 1;
    for (const string &rgb_file: rgb_files) {
        cv::Mat image = cv::imread(rgb_file);
        vector<cv::KeyPoint> keypoints;
        cv::Mat descriptor;
        detector->detectAndCompute(image, cv::Mat(), keypoints, descriptor);
        descriptors.push_back(descriptor);
        cout << "extracting features from image " << index++ << endl;
    }
    cout << "extract total " << descriptors.size() * 500 << " features." << endl;

    // create vocabulary
    cout << "creating vocabulary, please wait ... " << endl;
    DBoW3::Vocabulary vocab;
    vocab.create(descriptors);
    cout << "vocabulary info: " << vocab << endl;
    vocab.save("vocab_larger.yml.gz");
    cout << "done" << endl;

    return 0;
}
