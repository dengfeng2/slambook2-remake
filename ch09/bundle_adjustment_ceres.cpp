#include <iostream>
#include "common.h"
#include <ceres/ceres.h>
#include "SnavelyReprojectionError.h"

using namespace std;

string bal_file = "problem-16-22106-pre.txt";

void SolveBA(BALProblem &bal_problem);

int main(int argc, char *argv[]) {
    BALProblem bal_problem(bal_file);
    bal_problem.Normalize();
    bal_problem.Perturb(0.1, 0.5, 0.5);
    bal_problem.WriteToPLYFile("initial.ply");
    SolveBA(bal_problem);
    bal_problem.WriteToPLYFile("final.ply");
}

void SolveBA(BALProblem &bal_problem) {
    ceres::Problem problem;
    for (int i = 0; i < bal_problem.num_observations(); ++i) {
        const auto &observations = bal_problem.observations();

        // Each Residual block takes a point and a camera as input
        // and outputs a 2-dimensional Residual
        ceres::CostFunction *cost_function = SnavelyReprojectionError::Create(observations[2 * i + 0], observations[2 * i + 1]);

        // If enabled use Huber's loss function.
        ceres::LossFunction *loss_function = new ceres::HuberLoss(1.0);

        // Each observation corresponds to a pair of a camera and a point
        // which are identified by camera_index()[i] and point_index()[i]
        // respectively.

        auto &camera_param = bal_problem.mutable_camera_param();
        auto &world_points = bal_problem.mutable_world_points();
        double *camera = camera_param.data() + bal_problem.camera_index()[i] * 9;
        double *point = world_points.data() + bal_problem.point_index()[i] * 3;

        problem.AddResidualBlock(cost_function, loss_function, camera, point);
    }
    // show some information here ...
    std::cout << "bal problem file loaded..." << std::endl;
    std::cout << "bal problem have " << bal_problem.num_cameras() << " cameras and "
            << bal_problem.num_points() << " points. " << std::endl;
    std::cout << "Forming " << bal_problem.num_observations() << " observations. " << std::endl;

    std::cout << "Solving ceres BA ... " << endl;
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::LinearSolverType::SPARSE_SCHUR;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << "\n";
}
