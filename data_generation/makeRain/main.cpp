#include <iostream>
#include <random>
#include <glob.h>
#include <regex>
#include <boost/format.hpp>
#include <boost/filesystem.hpp>

#include "rain.h"

using namespace std;
namespace fs = boost::filesystem;
using boost::format;
using boost::str;

float SCALE = 0.25;

void getFiles(const string &pattern, vector<string> &filePath);

int main() {
    map<string, double> params;
    vector<string> imgPath;
    // getFiles("/mnt/d/Documents/Derain/Data/cityscapes/leftImage/"
            // "train/*/*.png", imgPath);
    getFiles("/mnt/d/Documents/Derain/Data/cityscapes/leftImg8bit_rain/"
        "train/*/*.png", imgPath);
    // getFiles("/media/ros/Workshop/ws/Datasets/cityscapes/leftImage/"
    //          "val/*/*.png", imgPath);

    //getFiles("~/repo/dataset/cityscapes/leftImage/"
    //         "test/*/*.png", imgPath);
    //getFiles("~/repo/dataset/cityscapes/leftImage/test/mainz/*.png", imgPath);

    // number of images each original image produces
    
    int totalIndex = imgPath.size(), numsPerImg{5};
    cout << "Total images: " << totalIndex << endl;

    // some random number generator
    mt19937 rng;
    rng.seed(random_device()());
    uniform_int_distribution<int> random_M(100, 500);
    uniform_int_distribution<int> random_B(4000, 8000);
    uniform_int_distribution<int> random_psi(80, 90);
    uniform_int_distribution<int> random_dia(7, 20);   // blur kernel size

    //create save directory
    string savePath{"/home/dixonhank/workspace/RaindropGen/_data/output/"};
    string gtPath = savePath + "gt";
    string RDPath =  savePath + "raindrop";
    string MaskPath =  savePath + "raindrop_mask";
    fs::create_directories(savePath);
    fs::create_directories(gtPath);
    fs::create_directories(RDPath);
    fs::create_directories(MaskPath);

    for(int index{0}; index < totalIndex; ++index) {
        fs::path img_path{imgPath[index]};        
        std::string img_name{img_path.filename().string()};
        
        if(index % 10 == 0) {
            cout << "Processing: " << setprecision(2) << index << " / " << totalIndex << " (" << float(index) / totalIndex << ")" << endl;
        }
        unsigned count{0};
        params["M"] = random_M(rng);
        params["B"] = random_B(rng);
        // params["psi"] = random_psi(rng);

        // params["M"] = 200;
        // params["B"] = 8000;
        params["psi"] = 90;
        
        Rain rain(params, imgPath[index]);
        cv::Mat img;
        cv::resize(rain.image, img, cv::Size(), SCALE, SCALE);
        cv::imwrite(str(format("%1%/%2%.png")%gtPath%img_name), img); 
        // cv::imshow("test_show input", img);

        for(int i{0}; i < numsPerImg; i++) {
            rain.render();      
            cv::Mat rain_img;
            cv::resize(rain.rain_image, rain_img, cv::Size(), SCALE, SCALE);
            // cv::imwrite(str(format("%1%/%2%_I.png")%savePath%index), img);
            // cv::imshow("test_show rain", rain_img);

            auto kernel = rain.get_kernel(random_dia(rng));
            rain.blur(kernel);

            // add foreground blur
            rain.blur_foreground(); //

            // cv::Mat mask, blur;
            // cv::resize(rain.mask, mask, cv::Size(), SCALE, SCALE, cv::INTER_NEAREST);
            // cv::resize(rain.blur_image, blur, cv::Size(), SCALE, SCALE);

            // cv::imwrite(str(format("%1%/%2%_%3%_M.png")%savePath%index%count), mask);
            // cv::imwrite(str(format("%1%/%2%_%3%_B.png")%savePath%index%count), blur);

            // std::string path_sem = std::regex_replace(imgPath[index], regex(R"(leftImage)"), "gtFine");
            // std::string path_sem_seg = std::regex_replace(path_sem, regex(R"(leftImg8bit)"), "gtFine_labelIds");
            // std::string path_ins_seg = std::regex_replace(path_sem, regex(R"(leftImg8bit)"), "gtFine_instanceIds");
            // std::string path_sem_seg_color = std::regex_replace(path_sem, regex(R"(leftImg8bit)"), "gtFine_color");
            
            // cv::Mat sem = cv::imread(path_sem_seg, -1);
            // cv::Mat sem_save;
            // cv::resize(sem, sem_save, cv::Size(), SCALE, SCALE, cv::INTER_NEAREST);
            // // cv::imwrite(str(format("%1%/%2%_%3%_S.png")%savePath%index%count), sem_save);
            // sem = cv::imread(path_sem_seg_color);
            // sem_save;
            // cv::resize(sem, sem_save, cv::Size(), SCALE, SCALE, cv::INTER_NEAREST);
            // // cv::imwrite(str(format("%1%/%2%_%3%_S_color.png")%savePath%index%count), sem_save);
            // sem = cv::imread(path_ins_seg, -1);
            // sem_save;
            // cv::resize(sem, sem_save, cv::Size(), SCALE, SCALE, cv::INTER_NEAREST);

            // cv::imwrite(str(format("%1%/%2%_%3%_Ins.png")%savePath%index%count), sem_save);
//            cv::imshow("test show mask", mask);
//            cv::imshow("test show blur", blur);
//
//            cv::waitKey();

            cv::imwrite(str(format("%1%/%2%_%3%.png")%RDPath%img_name%count), rain.blur_image);
            cv::imwrite(str(format("%1%/%2%_%3%.png")%MaskPath%img_name%count), rain.mask);
            ++count;
        }
    }
    return 0;
}

void getFiles(const string &pattern, vector<string> &filePath) {
    glob_t globBuf;

    glob(pattern.c_str(), GLOB_TILDE, NULL, &globBuf);
    
    for(unsigned i{0}; i < globBuf.gl_pathc; i++) {
        filePath.push_back(globBuf.gl_pathv[i]);
    }

    if(globBuf.gl_pathc > 0) {
        globfree(&globBuf);
    }
}

