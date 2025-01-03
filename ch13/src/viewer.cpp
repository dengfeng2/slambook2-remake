#include "viewer.h"

#include <pangolin/pangolin.h>
#include <opencv2/opencv.hpp>
#include <glog/logging.h>
#include <sophus/se3.hpp>
#include "map.h"
#include "frame.h"
#include "map_point.h"

namespace myslam {
    Viewer::Viewer(std::shared_ptr<Map> map) : map_(std::move(map)) {
        viewer_thread_ = std::thread(std::bind(&Viewer::ThreadLoop, this));
    }

    void Viewer::AddCurrentFrame(std::shared_ptr<Frame> frame) {
        std::unique_lock<std::mutex> lock(mutex_);
        current_frame_ = std::move(frame);
    }

    void Viewer::UpdateMap() {
        std::unique_lock<std::mutex> lock(mutex_);
        assert(map_ != nullptr);
        assert(map_ != nullptr);
        active_keyframes_ = map_->GetActiveKeyFrames();
        active_landmarks_ = map_->GetActiveLandmarks();
    }

    void Viewer::Close() {
        viewer_running_ = false;
        viewer_thread_.join();
    }

    void Viewer::ThreadLoop() {
        pangolin::CreateWindowAndBind("MySLAM", 1024, 768);
        glEnable(GL_DEPTH_TEST);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        pangolin::OpenGlRenderState vis_camera(
                pangolin::ProjectionMatrix(1024, 768, 400, 400, 512, 384, 0.1, 1000),
                pangolin::ModelViewLookAt(0, -5, -10, 0, 0, 0, 0.0, -1.0, 0.0));

        // Add named OpenGL viewport to window and provide 3D Handler
        pangolin::View &vis_display =
                pangolin::CreateDisplay()
                        .SetBounds(0.0, 1.0, 0.0, 1.0, -1024.0f / 768.0f)
                        .SetHandler(new pangolin::Handler3D(vis_camera));

        constexpr float green[3] = {0, 1, 0};

        while (!pangolin::ShouldQuit() && viewer_running_) {
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
            vis_display.Activate(vis_camera);

            std::unique_lock<std::mutex> lock(mutex_);
            if (current_frame_) {
                DrawFrame(current_frame_, green);
                Sophus::SE3d Twc = current_frame_->Pose().inverse();
                pangolin::OpenGlMatrix m(Twc.matrix());
                vis_camera.Follow(m, true);

                cv::Mat img = PlotFrameImage();
                cv::imshow("image", img);
                cv::waitKey(1);
            }

            if (map_) {
                DrawMapPoints();
            }

            pangolin::FinishFrame();
            usleep(5000);
        }

        LOG(INFO) << "Stop viewer";
    }

    void Viewer::DrawFrame(const std::shared_ptr<Frame> &frame, const float *color) {
        Sophus::SE3d Twc = frame->Pose().inverse();

        constexpr float sz = 1.0;
        constexpr int line_width = 2.0;
        constexpr float fx = 400;
        constexpr float fy = 400;
        constexpr float cx = 512;
        constexpr float cy = 384;
        constexpr float width = 1080;
        constexpr float height = 768;

        glPushMatrix();

        Sophus::Matrix4f m = Twc.matrix().cast<float>();
        glMultMatrixf((GLfloat *) m.data());

        if (color == nullptr) {
            glColor3f(1, 0, 0);
        } else
            glColor3f(color[0], color[1], color[2]);

        glLineWidth(line_width);
        glBegin(GL_LINES);
        glVertex3f(0, 0, 0);
        glVertex3f(sz * (0 - cx) / fx, sz * (0 - cy) / fy, sz);
        glVertex3f(0, 0, 0);
        glVertex3f(sz * (0 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
        glVertex3f(0, 0, 0);
        glVertex3f(sz * (width - 1 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
        glVertex3f(0, 0, 0);
        glVertex3f(sz * (width - 1 - cx) / fx, sz * (0 - cy) / fy, sz);

        glVertex3f(sz * (width - 1 - cx) / fx, sz * (0 - cy) / fy, sz);
        glVertex3f(sz * (width - 1 - cx) / fx, sz * (height - 1 - cy) / fy, sz);

        glVertex3f(sz * (width - 1 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
        glVertex3f(sz * (0 - cx) / fx, sz * (height - 1 - cy) / fy, sz);

        glVertex3f(sz * (0 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
        glVertex3f(sz * (0 - cx) / fx, sz * (0 - cy) / fy, sz);

        glVertex3f(sz * (0 - cx) / fx, sz * (0 - cy) / fy, sz);
        glVertex3f(sz * (width - 1 - cx) / fx, sz * (0 - cy) / fy, sz);

        glEnd();
        glPopMatrix();
    }

    void Viewer::DrawMapPoints() const {
        constexpr float red[3] = {1.0, 0, 0};
        for (auto &kf: active_keyframes_) {
            DrawFrame(kf.second, red);
        }

        glPointSize(2);
        glBegin(GL_POINTS);
        for (auto &landmark: active_landmarks_) {
            auto pos = landmark.second->Pos();
            glColor3f(red[0], red[1], red[2]);
            glVertex3d(pos[0], pos[1], pos[2]);
        }
        glEnd();
    }

    cv::Mat Viewer::PlotFrameImage() const {
        cv::Mat img_out;
        cv::cvtColor(current_frame_->left_img(), img_out, cv::COLOR_GRAY2RGB);
        for (size_t i = 0; i < current_frame_->left_features().size(); ++i) {
            if (current_frame_->left_features()[i]->map_point()) {
                auto feat = current_frame_->left_features()[i];
                cv::circle(img_out, feat->keypoint().pt, 2, cv::Scalar(0, 250, 0), 2);
            }
        }
        return img_out;
    }
}
