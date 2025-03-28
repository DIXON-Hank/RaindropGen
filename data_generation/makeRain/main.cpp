#include <iostream>
#include <random>
#include <glob.h>
#include <regex>
#include <chrono>
#include <boost/format.hpp>
#include <boost/filesystem.hpp>

#include "rain.h"

using namespace std;
namespace fs = boost::filesystem;
using boost::format;
using boost::str;

float SCALE = 0.5;

void getFiles(const string &pattern, vector<string> &filePath);
std::string get_current_time();

int main() {
    map<string, double> params;
    vector<string> imgPath;
    string time{get_current_time()}; //record generation time if needed
    getFiles("/mnt/d/Documents/Derain/Data/RDScityscapes/rainstreak/train/*.png", imgPath);
    // cout << imgPath[0] << endl;

    // number of images each original image produces
    int totalIndex = imgPath.size(), numsPerImg{1};
    cout << "Total images: " << totalIndex << endl;

    // some random number generator
    mt19937 rng;
    rng.seed(random_device()());
    uniform_int_distribution<int> random_M(100, 250); // glass distance (mm)
    uniform_int_distribution<int> random_B(8000, 15000); //background distance (mm)
    uniform_int_distribution<int> random_psi(45, 85); //glass angle
    uniform_int_distribution<int> random_dia(7, 20); // blur kernel size
    uniform_int_distribution<int> kernel_dist(19, 25); // gaussian kernel_size (blur area)
    uniform_int_distribution<int> blur_dist(9, 15); // gaussian blur kernel_size (blur intensity)
    int kernel_size{23}, blur_kernel_size{13};
    
    //create save directory
    string save_root{"/mnt/d/Documents/Derain/Data/RDScityscapes/"}; //remeber to add / in the end
    string gt_root = save_root + "gt/train";
    string foggy_root = save_root + "foggy/train";
    string all_root =  save_root + "all/train";
    string mask_root =  save_root + "raindrop_mask/train";
    string raindrop_root = save_root + "raindrop/train";
    fs::create_directories(save_root);
    fs::create_directories(gt_root);
    fs::create_directories(all_root);
    fs::create_directories(mask_root);
    fs::create_directories(raindrop_root);

    for(int index{0}; index < totalIndex; ++index) {
        fs::path img_path{imgPath[index]};        
        string img_name{img_path.stem().string()};
        if(index % 10 == 0) {
            cout << "Processing: " << setprecision(2) << index << " / " << totalIndex << " (" << float(index) / totalIndex << ")" << endl;
        }
        unsigned count{0};
        //Generation Parameters Settings
        params["M"] = random_M(rng);
        params["B"] = random_B(rng);
        params["psi"] = random_psi(rng);
        params["dropsize"] = 0.8;
        kernel_size = kernel_dist(rng) | 1;
        blur_kernel_size = blur_dist(rng) | 1;

        // params["M"] = 200;
        // params["B"] = 8000;
        // params["psi"] = 90;

        Rain all(params, imgPath[index], rng);
        string foggy_path = str(format("%1%/%2%_leftImg8bit_foggy_%3%.png")%foggy_root%all.frame_name%all.foggy_name);
        // cout << foggy_path << endl;
        Rain foggy(params, foggy_path, rng);
        // cout << foggy.frame_name << " " << foggy.foggy_name << endl;
        // cv::Mat img;
        // cv::resize(all.image, img, cv::Size(), SCALE, SCALE);
        // cv::imwrite(str(format("%1%/%2%.png")%gt_root%img_name), img); 

        for(int i{0}; i < numsPerImg; i++) {
            all.render();
            foggy.render();      
            cv::Mat all_img, raindrop_img;
            cv::resize(all.rain_image, all_img, cv::Size(), SCALE, SCALE);
            cv::resize(foggy.rain_image, raindrop_img, cv::Size(), SCALE, SCALE);
            auto kernel = all.get_kernel(random_dia(rng));
            all.blur(kernel);
            foggy.blur(kernel);

            // add foreground blur
            all.blur_foreground(kernel_size, blur_kernel_size); 
            foggy.blur_foreground(kernel_size, blur_kernel_size); 

            cv::Mat mask_img, all_blur, raindrop_blur;
            cv::resize(all.mask, mask_img, cv::Size(), SCALE, SCALE, cv::INTER_NEAREST);
            cv::resize(all.blur_image, all_blur, cv::Size(), SCALE, SCALE);
            cv::resize(foggy.blur_image, raindrop_blur, cv::Size(), SCALE, SCALE);
            cv::imwrite(str(format("%1%/%2%_%3%.png")%all_root%img_name%count), all_blur);
            cv::imwrite(str(format("%1%/%2%_%3%.png")%raindrop_root%img_name%count), raindrop_blur);
            cv::imwrite(str(format("%1%/%2%_%3%.png")%mask_root%img_name%count), mask_img);

//            cv::waitKey();
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

std::string get_current_time() {
    auto now = std::chrono::system_clock::now();            
    auto time_t_now = std::chrono::system_clock::to_time_t(now); 
    std::tm tm_now = *std::localtime(&time_t_now);          

    std::ostringstream oss;
    oss << std::put_time(&tm_now, "%Y-%m-%d_%H-%M-%S");     
    return oss.str();                                       
}


