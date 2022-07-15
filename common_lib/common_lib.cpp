#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <tensorflow/lite/c/c_api.h>


struct CIntArray {
  int length;
  int *data;
};

struct CUintArray {
  int length;
  unsigned int *data;
};

struct CFloatArray {
  int length;
  float *data;
};

struct CUcharArray {
  int length;
  unsigned char *data;
};

struct CBBox {
    float score;
    CFloatArray bbox; // 4
    CFloatArray landmarks; // 10
    CFloatArray label;
};

 struct Result {
     unsigned int length = 0;
     CBBox *boxes = NULL;
 };




extern "C" {
    float rect_intersection(float* a, float* b);
    float rect_union(float* a, float* b);
    float iou_score(float* a, float* b);
    unsigned int YUV2RGB(unsigned char y, unsigned char u, unsigned char v);
    CIntArray nms(CFloatArray bboxes, CFloatArray scores, double iou_th, double score_th);
    CFloatArray invoke(TfLiteInterpreter *interpreter, TfLiteTensor *in_tensor, CFloatArray in_data,
                       unsigned int out_shape);
    CFloatArray parseOutputData(TfLiteInterpreter *interpreter, unsigned int out_shape, unsigned int index);
    void initModel(const char *path, TfLiteInterpreter *&interpreter, TfLiteModel *&model,
                   TfLiteInterpreterOptions *&opt, TfLiteTensor *&input_ts);
    void cleanModel(TfLiteInterpreter *&interpreter, TfLiteModel *&model, TfLiteInterpreterOptions *&opt);
    CBBox parseBBox(float *scores, float *bboxes, float *landmarks, unsigned int i, unsigned int w, unsigned h);
    CFloatArray faceAlign2(CFloatArray img, CFloatArray bbox, CFloatArray landmarks, unsigned int height, unsigned int width);
    float *normalize(float *f1);

    class TFLModel {
    public:
        TfLiteInterpreter *detector, *extractor, *recognizer;
        TfLiteModel *det_model, *ext_model, *rec_model;
        TfLiteInterpreterOptions *det_opt, *ext_opt, *rec_opt;
        TfLiteTensor *det_input_ts, *ext_input_ts, *rec_input_ts;

        TFLModel(char *dir) {
            std::string fileNames[] = {"retinaface.tflite", "mb_face_net.tflite", "c_model.tflite"};
            std::string dir_s = dir;
            dir_s += "/";

            initModel((dir_s + fileNames[0]).c_str(), this->detector  ,this->det_model,this->det_opt,this->det_input_ts);
            initModel((dir_s + fileNames[1]).c_str(), this->extractor ,this->ext_model,this->ext_opt,this->ext_input_ts);
            initModel((dir_s + fileNames[2]).c_str(), this->recognizer,this->rec_model,this->rec_opt,this->rec_input_ts);
        }

        ~TFLModel() {
            cleanModel(this->detector  ,this->det_model,this->det_opt);
            cleanModel(this->extractor ,this->ext_model,this->ext_opt);
            cleanModel(this->recognizer,this->rec_model,this->rec_opt);
        }

        Result forward(CFloatArray img) {
            TfLiteTensorCopyFromBuffer(this->det_input_ts, img.data, img.length * sizeof(float));
            TfLiteInterpreterInvoke(this->detector);

            CFloatArray scores    = parseOutputData(this->detector, 16800 * 2, 0);
            CFloatArray landmarks = parseOutputData(this->detector, 16800 * 10, 1);
            CFloatArray bboxes    = parseOutputData(this->detector, 16800 * 4, 2);

            CIntArray idx = nms(bboxes, scores, 0.4, 0.4);

            std::vector<CBBox> ret_bboxes = {};

            for (unsigned int i = 0; i < idx.length; i++) {
                CBBox face = parseBBox(scores.data, bboxes.data, landmarks.data, idx.data[i],640,640);

                CFloatArray inputFace = faceAlign2(img, face.bbox, face.landmarks, 640, 640);
                CFloatArray f1 = invoke(this->extractor, this->ext_input_ts, inputFace, 128);
                f1.data = normalize(f1.data);
                face.label = invoke(this->recognizer, this->rec_input_ts, f1, 5);
                free(inputFace.data);
                free(f1.data);

                ret_bboxes.push_back(face);
            }

            Result ret;
            if (ret_bboxes.size() > 0){
                ret.length = ret_bboxes.size();
                size_t s = ret_bboxes.size() * sizeof(CBBox);
                ret.boxes = (CBBox *) malloc(s);
                memcpy(ret.boxes, ret_bboxes.data(), s);
            }

            free(scores.data);
            free(landmarks.data);
            free(bboxes.data);
            free(idx.data);

            return ret;
        }
    };




    __attribute__((visibility("default"))) __attribute__((used))
    void freeResult(Result res) {
        for (unsigned int i = 0; i < res.length; i++){
            free(res.boxes[i].bbox.data);
            free(res.boxes[i].landmarks.data);
            free(res.boxes[i].label.data);
        }
        free(res.boxes);
    }

    __attribute__((visibility("default"))) __attribute__((used))
    CBBox getBBox(CBBox *box) {
        return *box;
    }

    void initModel(const char *path, TfLiteInterpreter *&interpreter, TfLiteModel *&model,
                   TfLiteInterpreterOptions *&opt, TfLiteTensor *&input_ts){
        model = TfLiteModelCreateFromFile(path);
        opt = TfLiteInterpreterOptionsCreate();
        interpreter = TfLiteInterpreterCreate(model, opt);
        TfLiteInterpreterAllocateTensors(interpreter);
        input_ts = TfLiteInterpreterGetInputTensor(interpreter, 0);
    }

    void cleanModel(TfLiteInterpreter *&interpreter, TfLiteModel *&model, TfLiteInterpreterOptions *&opt){
        TfLiteInterpreterDelete(interpreter);
        TfLiteInterpreterOptionsDelete(opt);
        TfLiteModelDelete(model);
    }





    CFloatArray invoke(TfLiteInterpreter *interpreter, TfLiteTensor *in_tensor, CFloatArray in_data,
                       unsigned int out_shape){
         TfLiteTensorCopyFromBuffer(in_tensor, in_data.data, in_data.length * sizeof(float));
         TfLiteInterpreterInvoke(interpreter);
         return parseOutputData(interpreter, out_shape, 0);
    }

    CFloatArray parseOutputData(TfLiteInterpreter *interpreter, unsigned int out_shape, unsigned int index){
        CFloatArray out_data;
        out_data.length = out_shape;
        out_data.data = (float *) malloc(sizeof(float) * out_data.length);
        const TfLiteTensor *out_tensor = TfLiteInterpreterGetOutputTensor(interpreter, index);
        TfLiteTensorCopyToBuffer(out_tensor, out_data.data, sizeof(float) * out_shape);
        return out_data;
    }


    CBBox parseBBox(float *scores, float *bboxes, float *landmarks, unsigned int i, unsigned int width, unsigned height){
        float w = static_cast<float>(width);
        float h = static_cast<float>(height);
        CBBox ret;
        ret.score = scores[i * 2 + 1];
        float box[] = {
            bboxes[i * 4    ],
            bboxes[i * 4 + 1],
            bboxes[i * 4 + 2],
            bboxes[i * 4 + 3]
        };
        ret.bbox.length = 4;
        ret.bbox.data = (float *)(malloc(sizeof(float) * ret.bbox.length));
        memcpy(ret.bbox.data, box, sizeof(float) * ret.bbox.length);

        float landmark[] = {
            landmarks[i * 10    ] * w,
            landmarks[i * 10 + 1] * h,
            landmarks[i * 10 + 2] * w,
            landmarks[i * 10 + 3] * h,
            landmarks[i * 10 + 4] * w,
            landmarks[i * 10 + 5] * h,
            landmarks[i * 10 + 6] * w,
            landmarks[i * 10 + 7] * h,
            landmarks[i * 10 + 8] * w,
            landmarks[i * 10 + 9] * h,
        };
        ret.landmarks.length = 10;
        ret.landmarks.data = (float *)(malloc(sizeof(float) * ret.landmarks.length));
        memcpy(ret.landmarks.data, landmark, sizeof(float) * ret.landmarks.length);

        return ret;
    }











    // common
    __attribute__((visibility("default"))) __attribute__((used))
    CFloatArray createCFloatArray(unsigned int length, float *data) {
        CFloatArray arr;
        arr.length = length;
        arr.data = data;
        return arr;
    }

    __attribute__((visibility("default"))) __attribute__((used))
    CIntArray createCIntArray(unsigned int length, int *data) {
        CIntArray arr;
        arr.length = length;
        arr.data = data;
        return arr;
    }

    __attribute__((visibility("default"))) __attribute__((used))
    CUintArray createCUintArray(unsigned int length, unsigned int *data) {
        CUintArray arr;
        arr.length = length;
        arr.data = data;
        return arr;
    }

    __attribute__((visibility("default"))) __attribute__((used))
    CUcharArray createCUcharArray(unsigned int length, unsigned char *data) {
        CUcharArray arr;
        arr.length = length;
        arr.data = data;
        return arr;
    }



    // opencv 
    __attribute__((visibility("default"))) __attribute__((used))
    const char* opencv_version() {
        return CV_VERSION;
    }

    __attribute__((visibility("default"))) __attribute__((used))
    void process_image(char* inputPath, char* outputPath) {
        cv::Mat input = imread(inputPath, cv::IMREAD_GRAYSCALE);
        cv::Mat threshed, withContours;

        std::vector<std::vector<cv::Point>> contours;
        std::vector<cv::Vec4i> hierarchy;
        
        cv::adaptiveThreshold(input, threshed, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY_INV, 77, 6);
        cv::findContours(threshed, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_TC89_L1);
        
        cv::cvtColor(threshed, withContours, cv::COLOR_GRAY2BGR);
        cv::drawContours(withContours, contours, -1, cv::Scalar(0, 255, 0), 4);
        
        cv::imwrite(outputPath, withContours);
    }

    __attribute__((visibility("default"))) __attribute__((used))
    CIntArray nms(CFloatArray bboxes, CFloatArray scores, double iou_th, double score_th) {
        std::vector<int> proposal_idx;
        std::vector<float> proposal_scores;
        std::vector<float*> proposal_boxes;

        for (int i=0;i<16800;i++){
            float score = scores.data[i*2+1];
            if (score > score_th){
                proposal_idx.push_back(i);
                proposal_scores.push_back(score);
                float *box = (float *)(malloc(4 * sizeof(float)));
                box[0] = bboxes.data[i * 4];
                box[1] = bboxes.data[i * 4 + 1];
                box[2] = bboxes.data[i * 4 + 2];
                box[3] = bboxes.data[i * 4 + 3];
                proposal_boxes.push_back(box);
            }
        }

        // sort idx by score
        int n = proposal_idx.size();
        for (int i=0; i<(n-1);i++){
            for (int j=i+1;j<n;j++){
                float a = proposal_scores[i];
                float b = proposal_scores[j];
                if (a > b){
                    std::swap(proposal_idx[i],proposal_idx[j]);
                    std::swap(proposal_scores[i],proposal_scores[j]);
                    std::swap(proposal_boxes[i],proposal_boxes[j]);
                }
            }
        }

        std::vector<int> pick;
        int idx1, i;
        float *a, *b;
        float iou;
        while (proposal_idx.size() > 0){
            idx1 = proposal_idx.back();
            pick.push_back(idx1);
            proposal_idx.pop_back();

            a = proposal_boxes.back();
            proposal_boxes.pop_back();

            i = (int) proposal_idx.size()-1;
            while(i >= 0){
                b = proposal_boxes[i];
                iou = iou_score(a,b);
                if (iou > iou_th){
                    proposal_idx.erase(proposal_idx.begin()+i);
                    proposal_boxes.erase(proposal_boxes.begin()+i);
                    free(b);
                }
                i--;
            }
        }

        for (unsigned int i = 0; i < proposal_boxes.size(); i++){
            free(proposal_boxes[i]);
        }

        CIntArray ret_idx;
        ret_idx.length = pick.size();
        ret_idx.data = (int *) malloc(ret_idx.length * sizeof(int));
        memcpy(ret_idx.data, pick.data(), ret_idx.length * sizeof(int));

        return ret_idx;
    }


    float rect_intersection(float* a, float* b){
        float ax1 = a[0];
        float ay1 = a[1];
        float ax2 = a[2];
        float ay2 = a[3];
        float bx1 = b[0];
        float by1 = b[1];
        float bx2 = b[2];
        float by2 = b[3];

        float x1 = std::max(ax1, bx1);
        float y1 = std::max(ay1, by1);

        float x2 = std::min(ax2, bx2);
        float y2 = std::min(ay2, by2);


        float w = x2-x1;
        float h = y2-y1;

        float i;
        if (w<0 || h<0)
            i = 0;
        else
            i = w*h;

        return i;
    }

    float rect_union(float* a, float* b){
        float ax1 = a[0];
        float ay1 = a[1];
        float ax2 = a[2];
        float ay2 = a[3];
        float bx1 = b[0];
        float by1 = b[1];
        float bx2 = b[2];
        float by2 = b[3];


        float x1 = std::min(ax1, bx1);
        float y1 = std::min(ay1, by1);

        float x2 = std::max(ax2, bx2);
        float y2 = std::max(ay2, by2);

        float w = x2-x1;
        float h = y2-y1;

        float u = w*h;
        return u;
    }

    float iou_score(float* a, float* b){
        float i = rect_intersection(a,b);
        float u = rect_union(a,b);
        float iou = i/u;

        return iou;
    }



    __attribute__((visibility("default"))) __attribute__((used))
    CFloatArray faceAlign(CUintArray img, CFloatArray bbox, CFloatArray landmarks, unsigned int height, unsigned int width) {
        unsigned char *img2 = (unsigned char *)(malloc(height*width*4));
        for (int i=0;i<(height*width);i++){
            img2[i*4]   = img.data[i];
            img2[i*4+1] = img.data[i] >> 8 ;
            img2[i*4+2] = img.data[i] >> 16;
            img2[i*4+3] = img.data[i] >> 24;
        }

        cv::Mat img3(height,width,CV_8UC4,img2);

        std::vector<cv::Point2f> src{
            cv::Point2f( landmarks.data[0], landmarks.data[1] ),
            cv::Point2f( landmarks.data[2], landmarks.data[3] ),
            cv::Point2f( landmarks.data[4], landmarks.data[5] ),
            cv::Point2f( landmarks.data[6], landmarks.data[7] ),
            cv::Point2f( landmarks.data[8], landmarks.data[9] )
        };


        std::vector<cv::Point2f> dst{
            cv::Point2f( 38.2946  , 51.6963 ),
            cv::Point2f( 73.5318  , 51.5014 ),
            cv::Point2f( 56.0252  , 71.7366 ),
            cv::Point2f( 41.5493  , 92.3655 ),
            cv::Point2f( 70.729904, 92.2041 )
        };



        cv::Mat warp_dst, img_rgb, img_f;
        cv::Size im_size(112,112);

        cv::Mat warp_mat = cv::estimateAffinePartial2D( src, dst );
        cv::warpAffine( img3, warp_dst, warp_mat, im_size );

        cv::cvtColor(warp_dst, img_rgb, cv::COLOR_RGBA2RGB);
        img_rgb.convertTo(img_f, CV_32FC3);

        float *ret = (float *)(malloc(sizeof(float) * 112*112*3));
        memcpy(ret, img_f.data, sizeof(float) * 112*112*3);

        CFloatArray ali_face;
        ali_face.length = 112*112*3;
        ali_face.data = ret;

        free(img2);

        return ali_face;
    }


    __attribute__((visibility("default"))) __attribute__((used))
    CFloatArray faceAlign2(CFloatArray img, CFloatArray bbox, CFloatArray landmarks, unsigned int height, unsigned int width) {
        cv::Mat img3(height,width,CV_32FC3,img.data);

        std::vector<cv::Point2f> src{
            cv::Point2f( landmarks.data[0], landmarks.data[1] ),
            cv::Point2f( landmarks.data[2], landmarks.data[3] ),
            cv::Point2f( landmarks.data[4], landmarks.data[5] ),
            cv::Point2f( landmarks.data[6], landmarks.data[7] ),
            cv::Point2f( landmarks.data[8], landmarks.data[9] )
        };

        std::vector<cv::Point2f> dst{
            cv::Point2f( 38.2946  , 51.6963 ),
            cv::Point2f( 73.5318  , 51.5014 ),
            cv::Point2f( 56.0252  , 71.7366 ),
            cv::Point2f( 41.5493  , 92.3655 ),
            cv::Point2f( 70.729904, 92.2041 )
        };

        cv::Mat warp_dst;
        cv::Size im_size(112,112);

        cv::Mat warp_mat = cv::estimateAffinePartial2D( src, dst );
        cv::warpAffine( img3, warp_dst, warp_mat, im_size );

        float *ret = (float *)(malloc(sizeof(float) * 112*112*3));
        memcpy(ret, warp_dst.data, sizeof(float) * 112*112*3);

        CFloatArray ali_face;
        ali_face.length = 112*112*3;
        ali_face.data = ret;

        return ali_face;
    }


     __attribute__((visibility("default"))) __attribute__((used))
     float *normalize(float *f1) {
        cv::Mat f3 = cv::Mat(1,128,CV_32FC1,f1);
        cv::normalize(f3,f3);
        float *ret = (float *)(malloc(sizeof(float) * 128));
        memcpy(ret, f3.data, sizeof(float) * 128);
        return ret;
    }


    __attribute__((visibility("default"))) __attribute__((used))
    CFloatArray convertYUVtoRGB(unsigned char *yArr, unsigned char *uArr, unsigned char *vArr,
             int yRowStride, int uvRowStride, int uvPixelStride,
             unsigned int width, unsigned int height, unsigned int rotateAngle, bool flip) {

        unsigned int yOffset,uvOffset;
        unsigned char y,u,v;

        unsigned int size = width*height;

        unsigned int *pixels;
        pixels = (unsigned int *) malloc(size * sizeof(unsigned int));

        for (int row=0; row<height; row++) {
            for (int col=0; col<width; col++) {
                yOffset  = row * yRowStride + col;
                uvOffset = ((row/uvPixelStride * uvRowStride/uvPixelStride) + col/uvPixelStride)*uvPixelStride;

                y = (unsigned char)yArr[yOffset];
                u = (unsigned char)uArr[uvOffset];
                v = (unsigned char)vArr[uvOffset];
                pixels[row*width+col] = YUV2RGB(y,u,v);
            }
        }

        unsigned int x1,y1,len;
        if (width<height) {
            len = width;
            x1 = 0;
            y1 = (height-width)/2;
        } else {
            len = height;
            x1 = (width-height)/2;
            y1 = 0;
        }

        cv::Mat img(height,width,CV_8UC4,pixels);
        cv::Rect roi(x1,y1,len,len);
        cv::Mat crop;

        if (rotateAngle == 0){
            img(roi).copyTo(crop);
        }
        if (rotateAngle == 90){
            cv::rotate(img(roi), crop, cv::ROTATE_90_CLOCKWISE);
        }
        if (rotateAngle == 180){
            cv::rotate(img(roi), crop, cv::ROTATE_180);
        }
        if (rotateAngle == 270){
            cv::rotate(img(roi), crop, cv::ROTATE_90_COUNTERCLOCKWISE);
        }

        if (flip){
            cv::Mat dst;
            cv::flip(crop, dst, 1);
            crop = dst;
        }

        free(pixels);

//        unsigned int *ret = (unsigned int *)(malloc(sizeof(unsigned int) * len*len));
//        memcpy(ret, crop.data, sizeof(unsigned int) * len*len);
//
//        CUintArray crop_img;
//        crop_img.length = len*len;
//        crop_img.data = ret;
//
//        return crop_img;


        cv::Mat s_channels[4];
        cv::split(crop, s_channels);
        cv::Mat img1(len,len,CV_8UC3);
        cv::Mat m_channels[3] = {s_channels[2],s_channels[1],s_channels[0]};
        cv::merge(m_channels, 3, img1);

        cv::Mat img2(640,640,CV_8UC3);
        cv::resize(img1, img2, cv::Size(640,640), cv::INTER_LINEAR);
//        cv::imwrite("/data/user/0/com.example.face_recognition/app_flutter/test.jpg", img2);

        cv::Mat img3;
        img2.convertTo(img3, CV_32FC3);

        float *ret = (float *)(malloc(sizeof(float) * 640*640*3));
        memcpy(ret, img3.data, sizeof(float) * 640*640*3);

        CFloatArray crop_img;
        crop_img.length = 640*640*3;
        crop_img.data = ret;

        return crop_img;
    }

    __attribute__((visibility("default"))) __attribute__((used))
    unsigned int YUV2RGB(unsigned char Y, unsigned char U, unsigned char V){
        float u = (float)U - 128;
        float v = (float)V - 128;
        float y = (float)Y * 1.164;

        unsigned int r = (unsigned int)(y + 2.018 * v);
        unsigned int g = (unsigned int)(y - 0.813 * v - 0.391 * u);
        unsigned int b = (unsigned int)(y + 1.596 * u);

        if (r < 0) r = 0;
        if (g < 0) g = 0;
        if (b < 0) b = 0;
        if (r > 255) r = 255;
        if (g > 255) g = 255;
        if (b > 255) b = 255;

        return 0xff000000|(0x00ff0000&(r<<16))|(0x0000ff00&(g<<8))|(0x000000ff&b);
    }



    // tflite
    __attribute__((visibility("default"))) __attribute__((used))
    const char* tflite_version() {
        return TfLiteVersion();
    }

    __attribute__((visibility("default"))) __attribute__((used))
    void *createModel(char *dir) {
        TFLModel* model = new TFLModel(dir);
        return (void *)model;
    }

    __attribute__((visibility("default"))) __attribute__((used))
    Result forwardModel(void *model_p, CFloatArray input) {
        TFLModel* model = (TFLModel *) model_p;
        return model->forward(input);
    }

    __attribute__((visibility("default"))) __attribute__((used))
    void deleteModel(void *model_p) {
        TFLModel* model = (TFLModel *) model_p;
        delete model;
    }



}














