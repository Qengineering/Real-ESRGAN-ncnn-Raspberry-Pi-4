// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

// modified 12-31-2022 Q-engineering

#include "layer.h"
#include "net.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
#include <vector>
#include <iostream>

#include "realesrgan.h"

int main(int argc, char** argv)
{
    std::chrono::steady_clock::time_point Tbegin, Tend;

    if (argc != 2)
    {
        fprintf(stderr, "Usage: %s [imagepath]\n", argv[0]);
        return -1;
    }

    RealESRGAN real_net;
    real_net.load("./real_esrgan.param", "./real_esrgan.bin");

    const char* imagepath = argv[1];

    cv::Mat m = cv::imread(imagepath, 1);
    if (m.empty())
    {
        fprintf(stderr, "cv::imread %s failed\n", imagepath);
        return -1;
    }

    std::cout << "Start upsampling ...." << std::endl;
    Tbegin = std::chrono::steady_clock::now();

    cv::Mat img_up;
    real_net.tile_process(m, img_up);

    Tend = std::chrono::steady_clock::now();

    //calculate inference time
    int f = std::chrono::duration_cast <std::chrono::milliseconds> (Tend - Tbegin).count();
    std::cout << "Inference time : " << f/1000.0 <<  " Sec" << std::endl;

    cv::imshow("RPi4 - 1.95 GHz - 2 GB ram",img_up);
    cv::imwrite("test.jpg",img_up);
    cv::waitKey();

    return 0;
}
