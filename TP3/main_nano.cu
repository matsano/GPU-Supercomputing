#include "mylib.cuh"
#include "mylib.h"
#include <cuda_runtime.h>

// acces au flux de la camera
std::string gstreamer_pipeline(int capture_width, int capture_height,
                               int display_width, int display_height,
                               int framerate, int flip_method) {
  return "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)" +
         std::to_string(capture_width) + ", height=(int)" +
         std::to_string(capture_height) +
         ", format=(string)NV12, framerate=(fraction)" +
         std::to_string(framerate) +
         "/1 ! nvvidconv flip-method=" + std::to_string(flip_method) +
         " ! video/x-raw, width=(int)" + std::to_string(display_width) +
         ", height=(int)" + std::to_string(display_height) +
         ", format=(string)BGRx ! videoconvert ! video/x-raw, "
         "format=(string)BGR ! appsink";
}

int main(int, char **) {
  int capture_width = 1280;
  int capture_height = 720;
  int display_width = 640;
  int display_height = 360;
  int framerate = 60;
  int flip_method = 0;
  int c = '1';

  std::string pipeline =
      gstreamer_pipeline(capture_width, capture_height, display_width,
                         display_height, framerate, flip_method);
  std::cout << "Using pipeline: \n\t" << pipeline << "\n";

  cv::VideoCapture cap(pipeline, cv::CAP_GSTREAMER);

  if (!cap.isOpened()) // check if we succeeded
    return -1;

  while (1) {
    Mat frame;
    cap >> frame;
    int c_new;
    c_new = cv::waitKey(10);
    if (c_new != -1)
      c = c_new;

    switch (c) {
    case '1': {
      imshow("frame", frame);
      break;
    }
    case '2': {
      Mat NB = noirBlanc(frame);
      imshow("NoirEtBlanc", NB);
      break;
    }
    case '3': {
      Mat seuil = seuillage(frame);
      imshow("seuillage", seuil);
      break;
    }
    case '4': {
      Mat cont = contour(frame);
      imshow("contour", cont);
      break;
    }
    case '5': {
      Mat seuilgpu = seuillageGPU(frame);
      imshow("seuillage GPU", seuilgpu);
      break;
    }
    case '6': {
      Mat sobelgpu = sobelGPU(frame);
      imshow("Sobel GPU", sobelgpu);
      break;
    }

    case '7': {
      Mat nbgpu = nbGPU(frame);
      imshow("NB GPU", nbgpu);
      break;
    }

    case '0': {
      destroyAllWindows();
      break;
    }
    default:
      break;
    }

    if (c == '\e')
      break;
  }
  // When everything done, release the video capture object
  cap.release();

  // Closes all the frames
  destroyAllWindows();

  return 0;
}
