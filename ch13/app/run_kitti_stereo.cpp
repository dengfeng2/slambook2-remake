#include <memory>

#include "visual_odometry.h"

int main(int argc, char *argv[]) {
    auto vo = myslam::VisualOdometry::Create("/home/hdf/workspace/Kitti/dataset/sequences/05");
    assert(vo->Init());
    vo->Run();
    return 0;
}
