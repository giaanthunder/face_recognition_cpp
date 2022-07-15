#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include "common_lib/common_lib.cpp"

int main(){
    char dir[] = "assets";
    TFLModel *models = (TFLModel*) createModel(dir);

    cv::Mat img, img1;
    CFloatArray img2;

    // cv::VideoCapture cap;
    // cap.open(0, cv::CAP_ANY);      // 0 = autodetect default API
    // if (!cap.isOpened()) {
    //     return -1;
    // }
    // while (1) {
    //     cap.read(img);
    //     if (img.empty())
    //         break;


        img = cv::imread("hl_ngan.jpg");
        img.convertTo(img1, CV_32FC3);
        cv::cvtColor(img1, img1, cv::COLOR_BGR2RGB);
        img2.length = img1.rows * img1.cols * 3;
        img2.data = (float*) malloc(img2.length * sizeof(float));
        memcpy(img2.data, img1.data, img2.length * sizeof(float));
        Result ret = forwardModel(models, img2);
        CBBox *p = ret.boxes;
        for (int i=0; i<ret.length; i++){
            CFloatArray bbox = p[i].bbox;

            float x1 = bbox.data[0]*640;
            float y1 = bbox.data[1]*640;
            float x2 = bbox.data[2]*640;
            float y2 = bbox.data[3]*640;

            cv::rectangle(img, cv::Point(x1,y1), cv::Point(x2,y2), cv::Scalar(0,255,0));
        }

        cv::imshow("Live", img);
        cv::waitKey();
        free(img2.data);
        freeResult(ret);

    //     if (cv::waitKey('q') >= 0)
    //         break;
    // }


    return 0;
}






