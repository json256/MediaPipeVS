#pragma once

#include <cstdlib>

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/port/commandlineflags.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/opencv_highgui_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/opencv_video_inc.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/formats/landmark.pb.h"


::mediapipe::StatusOrPoller InitGraph(std::string calculator_graph_config_contents);

::mediapipe::Status RunGraph();

::mediapipe::Status CloseGraph();

::mediapipe::Status ProcessFrame(mediapipe::OutputStreamPoller& poller, cv::Mat& camera_frame, std::vector<mediapipe::NormalizedLandmarkList>& vec_landmarks);
