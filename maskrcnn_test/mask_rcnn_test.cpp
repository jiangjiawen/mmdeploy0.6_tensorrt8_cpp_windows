//
// Created by Administrator on 9/15/2022.
//
#include "maskrcnn.h"

#include <chrono>
using namespace std::literals;

std::vector<cv::Scalar> colors{cv::Scalar(139,0,139), cv::Scalar(210,105,30), cv::Scalar(112,128,144), cv::Scalar(100, 149, 237), cv::Scalar(176,224,230)};

int main(){
    SampleMaskRCNN sample("res18_maskrcnn.engine");
    sample.InitInfo();

    const std::chrono::time_point<std::chrono::steady_clock> start =
            std::chrono::steady_clock::now();

    auto output_buffer_label = std::unique_ptr<int>{ new int[sample.get_outputsize_labels()] };
    auto output_buffer_dets = std::unique_ptr<float>{ new float[sample.get_outputsize_dets()] };
    auto output_buffer_masks = std::unique_ptr<float>{ new float[sample.get_outputsize_masks()] };

    gLogInfo << "Running TensorRT inference for maskrcnn sperm xc" << std::endl;
    if (!sample.infer("01414.jpg", output_buffer_label, output_buffer_dets, output_buffer_masks))
    {
        return -1;
    }

    const auto pred_label = output_buffer_label.get();
    const auto pred_bbox = output_buffer_dets.get();
    const auto pred_mask = output_buffer_masks.get();

    const auto end = std::chrono::steady_clock::now();

    std::cout
            << "model takes "
            << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << "µs ≈ "
            << (end - start) / 1ms << "ms ≈ "
            << (end - start) / 1s << "s.\n";

    //画图片很花时间
    const cv::Mat img = cv::imread("01414.jpg");

    float SCORE_THRESHOLD=0.45;

    offset off=sample.GetOffsetInfo();
    const int MASK_STEP_SIZE = 28*28;
    for(int ind=0;ind<100;ind++) {
        float score = *(pred_bbox + ind * 5 + 4);
        if(score < SCORE_THRESHOLD)
            continue;

        float x0 = *(pred_bbox + ind * 5);
        float y0 = *(pred_bbox + ind * 5 + 1);
        float x1 = *(pred_bbox + ind * 5 + 2);
        float y1 = *(pred_bbox + ind * 5 + 3);

        std::cout << score << " " << x0 << " " << y0 << " " << x1 << " " << y1 << std::endl;

        int x0_t = std::max(int((x0-off.left)/off.scala_factor), 0);
        int y0_t = std::max(int((y0-off.top)/off.scala_factor), 0);
        int x1_t = std::min(int((x1-off.left)/off.scala_factor), img.cols);
        int y1_t = std::min(int((y1-off.top)/off.scala_factor), img.rows);

        cv::rectangle(img, cv::Point(x0_t, y0_t), cv::Point(x1_t, y1_t), cv::Scalar(255,255,0), 2);
        cv::putText(img, std::to_string(*(pred_label+ind)), cv::Point(x0_t, y0_t),
                    cv::FONT_HERSHEY_COMPLEX, 1, cv::Scalar(0, 0, 255), 1, 1, false);

        float* mask_ind = pred_mask + MASK_STEP_SIZE * ind ;
        cv::Mat r_mat_(28, 28, CV_32FC1, mask_ind);
        cv::Mat r_mat;
        r_mat_.convertTo(r_mat, CV_8UC1);


        cv::Mat resize_back;
        cv::resize(r_mat, resize_back, cv::Size(int(x1_t-x0_t+1),int(y1_t-y0_t+1)));
        std::vector<std::vector<cv::Point> > contours;
        cv::findContours(resize_back, contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
        cv::Scalar color = colors[*(pred_label+ind)];
        cv::drawContours(img, contours, -1, color, 1, 1,cv::noArray(), INT_MAX, cv::Point(x0_t , y0_t));
    }

    cv::imwrite("result.png", img);

    return 0;
}
