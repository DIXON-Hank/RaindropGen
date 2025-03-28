#include "rain.h"
#include <cmath>
#include <regex>
#include <jsoncpp/json/json.h>
#include <random>

//#pragma clang diagnostic push
//#pragma clang diagnostic ignored "-Wunused-variable"
using namespace std;
using namespace arma;

string camera_root = "/mnt/d/Documents/Derain/Data/mydataset/camera";

Rain::Rain(map<string, double> params, string image_path, const mt19937 seed) {
    rng = seed;
    M = params["M"];
    B = params["B"];
    dropsize = params["dropsize"];
    psi = params["psi"] / 180.0 * M_PI;
    gamma = asin(n_air / n_water);
    image = cv::imread(image_path);
    frame_name = extract_name(image_path)[0];
    city_name = extract_name(image_path)[1];
    foggy_name = extract_name(image_path)[2];
    intrinsic = get_intrinsic();
    normal = Row<double> {0.0, -1.0 * cos(psi), sin(psi)};
    o_g = (normal(2) * M / dot(normal, normal)) * normal;
}

vector<string> Rain::extract_name(const string &path) {
    smatch match_frame, match_city, match_foggy; // get frame_name and city_name from image_path
    if (!regex_search(path, match_frame, regex(R"(\w+_\d+_\d+)"))){
        throw runtime_error("Framename extraction failed!!");
    }
    if (!regex_search(path, match_city, regex(R"(\w+(?=_\d+_\d+))"))){
        throw runtime_error("Cityname extraction failed!!");
    }
    if (!regex_search(path, match_foggy, regex(R"(beta_[0-9]+(?:\.[0-9]+)?)"))){
        throw runtime_error("Foggyname extraction failed!!");
    }
    // since there is no matching foggy image for beta=0.015, hereby use beta=0.02 instead 
    if (match_foggy[0].str() == "beta_0.01" || match_foggy[0].str() == "beta_0.005"){
        return {match_frame[0].str(), match_city[0].str(), match_foggy[0].str()}; 
    }
    else if (match_foggy[0].str() == "beta_0.015" || match_foggy[0].str() == "beta_0.02"){
        return {match_frame[0].str(), match_city[0].str(), "beta_0.02"};
    }
    else 
        return {"e", "e", "e"};
}

Mat<double> Rain::get_intrinsic() {
    // get intrinstic file path
    string json_path;
    json_path = camera_root + "/" + city_name + "/" + frame_name + "_camera.json";

    // cout << json_path << endl;
    ifstream stream(json_path, ifstream::binary); 

    Json::Value root;
    stream >> root;
    root = root.get("intrinsic", 0);
    intrinsic = zeros<mat>(3, 3);
    intrinsic(0, 0) = root.get("fx", 0).asDouble();
    intrinsic(1, 1) = root.get("fy", 0).asDouble();
    intrinsic(2, 2) = 1.0;
    intrinsic(0, 2) = root.get("u0", 0).asDouble();
    intrinsic(1, 2) = root.get("v0", 0).asDouble();

    return intrinsic;
}

Row<double> Rain::to_glass(const double &x, const double &y) {
    double w = M * tan(psi) / (tan(psi) - (y - intrinsic(1, 2)) / intrinsic(1, 1));
    double u = w * (x - intrinsic(0, 2)) / intrinsic(0, 0);
    double v = w * (y - intrinsic(1, 2)) / intrinsic(1, 1);

    return Row<double> {u, v, w};
}

double Rain::w_in_plane(const double &u, const double &v) {
    return (normal(2)*M - normal(0)*u - normal(1)*v) / normal(2);
}

void Rain::get_sphere_raindrop(const int &W, const int &H) {
    g_centers.clear();
    g_radius.clear();
    centers.clear();
    radius.clear();

    //raindrop location
    auto left_upper = to_glass(0, 0);
    auto left_bottom = to_glass(0, H);
    auto right_upper = to_glass(W, 0);
    auto right_bottom = to_glass(W, H);

    uniform_int_distribution<mt19937::result_type> random_rain(50, 200); // raindrop number
    uniform_int_distribution<mt19937::result_type> random_tau(30, 45); //incident angle of each raindrop
    uniform_real_distribution<double> random_loc(0.0, 1.0);

    int n = random_rain(rng);
    int max_attempts = 100; // Maximum attempts to find a non-overlapping position
    
    for(int i = 0; i < n; i++) {
        bool found_valid = false;
        for (int attempt = 0; attempt < max_attempts; attempt++) {
            double u = left_bottom(0) + (right_bottom(0) - left_bottom(0)) * random_loc(rng);
            double v = left_upper(1) + (right_bottom(1) - left_upper(1)) * random_loc(rng);
            double w = w_in_plane(u, v);

            double tau = random_tau(rng);
            tau  = tau / 180 * M_PI;

            double glass_r = dropsize * (1 + 1 * random_loc(rng)); //raindrop size
            double r_sphere = glass_r / sin(tau);

            Row<double> g_c  {u, v, w};
            Row<double> c = g_c - normal * r_sphere * cos(tau);
            
            // Check for overlap with existing raindrops
            bool overlap = false;
            for (size_t j = 0; j < g_centers.size(); j++) {
                double dist = norm(g_c - g_centers[j]);
                if (dist < g_radius[j] + glass_r) {
                    overlap = true;
                    // cout << "Overlap Found" << endl;
                    break;
                }
            }

            // If no overlap, accept this raindrop
            if (!overlap) {
                g_centers.push_back(std::move(g_c));
                g_radius.push_back(glass_r);
                centers.push_back(std::move(c));
                radius.push_back(r_sphere);
                found_valid = true;
                break;
            }
        }        
    }
}


int Rain::in_sphere_raindrop(const int &x, const int &y) {
    auto p = to_glass(x, y);

    for(uint i = 0; i < g_centers.size(); i++) {
        if(norm(p - g_centers[i]) <= g_radius[i])
            return i;
    }
    return -1;
}

/**
 * Using the sphere section model
 * @param x
 * @param y
 * @param id    The id of rain drops
 * @return
 */
Row<double> Rain::to_sphere_section_env(const int &x, const int &y, const int &id) {
    Row<double> center = centers[id];
    double r_sphere = radius[id];

    Row<double> p_g = to_glass(x, y);

    double alpha = acos(dot(p_g, normal) / norm(p_g));
    double beta = asin(n_air * sin(alpha) / n_water);

    Row<double> po = p_g - o_g;
    po = po / norm(po);
    Row<double> i_1 = normal + tan(beta) * po;
    i_1 = i_1 / norm(i_1);

    Row<double> oc = p_g - center;
    double tmp = dot(i_1, oc);
    double d = -(tmp) + sqrt(pow(tmp, 2.0) - dot(oc, oc) + pow(r_sphere, 2.0));
    Row<double> p_w = p_g + d * i_1;

    Row<double> normal_w = p_w - center;
    normal_w = normal_w / norm(normal_w);

    d = (dot(p_w, normal_w) - dot(normal_w, p_g)) / dot(normal_w, normal_w);
    Row<double> p_a = p_w - (d * normal_w + p_g);
    p_a = p_a / norm(p_a);

    double eta = acos(dot(normal_w, p_w - p_g) / norm(p_w - p_g));
    if(eta >= gamma)
        throw "total refrection";

//    std::cout << "eta: " << eta << ", gamma: " << gamma << " (" << x << ", " << y << "), id: " << id << std::endl;

    double theta = asin(n_water * sin(eta) / n_air);
    Row<double> i_2 = normal_w + tan(theta) * p_a;

    Row<double> p_e = p_w + ((B - p_w[2]) / i_2[2]) * i_2;
    Row<double> p_i = trans(intrinsic * trans(p_e) / B);
    p_i = round(p_i);
    return p_i;
}

/**
 * Render the rain-drop image
 */
void Rain::render(const std::string mode) {
    int h = image.rows;
    int w = image.cols;
    rain_image = image.clone();
    mask = cv::Mat(h, w, CV_8UC1, cv::Scalar(0));
    get_sphere_raindrop(w, h);
    Row<double> p;
    for(int x = 0; x < w; x++) {
        for(int y = 0; y < h; y++) {
            int i;
            i = in_sphere_raindrop(x, y);
            if(i != -1) {
                try {
                    p = to_sphere_section_env(x, y, i);
                } catch (const char* msg) {
                    rain_image.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 0);
                    mask.at<char>(y, x) = (char)255;
                    continue;
                }
                int u = p(0), v = p(1);
                if(u >= w)
                    u = w - 1;
                else if(u < 0)
                    u = 0;

                if(v >= h)
                    v = h - 1;
                else if(v < 0)
                    v = 0;

                rain_image.at<cv::Vec3b>(y, x) = image.at<cv::Vec3b>(v, u);
                mask.at<char>(y, x) = char(255);
            }
        }
    }
}

cv::Mat Rain::get_kernel(int diameter) {
    cv::Mat kernel(diameter, diameter, CV_32FC1, cv::Scalar(0.0));
    double radius = diameter / 2.0;
    int count = 0;
    for(int i = 0;  i < diameter; i++) {
        for(int j = 0; j < diameter; j++) {
            if(pow(i - radius, 2.0) + pow(j - radius, 2.0) < pow(radius, 2.0)) {
                kernel.at<float>(i, j) = 1.0;
                ++count;
            }
        }
    }
    kernel /=  count;
    return kernel;
}

void Rain::blur(const cv::Mat &kernel) {
    blur_image = rain_image.clone();
    // blur_image = rain.clone();
    // cv::Mat blured;
    // cv::filter2D(rain_image, blured, -1, kernel);
    
    // blured = blured * 1.2;
    // blured.copyTo(blur_image, mask);
}

/**
 * @param kernel_size Size of the dilation kernel. 
 *                    - Controls the expansion of the blurred region.
 *                    - Larger values expand the mask further, increasing the size of the blurred area
 * @param blur_kernel_size Size of the Gaussian blur kernel. 
 *                         - Controls the intensity of the blur effect.
 *                         - Larger values result in stronger blur and a smoother appearance
 */
void Rain::blur_foreground(const int &kernel_size, const int &blur_kernel_size) { //TODO add random kernel_size
    cv::Mat dilated_mask; 
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(kernel_size, kernel_size));
    cv::dilate(mask, dilated_mask, kernel);
    // generate smooth mask
    cv::Mat smooth_mask;
    cv::GaussianBlur(dilated_mask, smooth_mask, cv::Size(kernel_size, kernel_size), blur_kernel_size / 6.0);
    smooth_mask.convertTo(smooth_mask, CV_32FC3, 1.0 / 255.0); // normalize to [0, 1]
    // bluring on whole image
    cv::Mat blur_foreground;
    cv::GaussianBlur(blur_image, blur_foreground, cv::Size(blur_kernel_size, blur_kernel_size), blur_kernel_size / 6.0);
    
    // Blending, first convert to 32 bit
    cv::Mat blended_image;
    cv::Mat blur_image_f, blur_foreground_f;
    blur_image.convertTo(blur_image_f, CV_32FC3);
    blur_foreground.convertTo(blur_foreground_f, CV_32FC3);
    cv::Mat smooth_mask_3c;
    cv::cvtColor(smooth_mask, smooth_mask_3c, cv::COLOR_GRAY2BGR);

    cv::Mat ones_single = cv::Mat::ones(smooth_mask_3c.size(), CV_32F), ones_3c;
    cv::cvtColor(ones_single, ones_3c, cv::COLOR_GRAY2BGR);
    blended_image = blur_image_f.mul(ones_3c - smooth_mask_3c) + blur_foreground_f.mul(smooth_mask_3c);
    blended_image.convertTo(blur_image, CV_8UC3); // covert back to 8 bit
}

#pragma clang diagnostic pop