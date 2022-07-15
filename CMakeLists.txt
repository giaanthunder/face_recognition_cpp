

cmake_minimum_required(VERSION 3.4.1)
project(FaceRecognition)
set(CMAKE_CXX_STANDARD 11)
set(LIB_DIR ${PROJECT_SOURCE_DIR}/jniLibs/ubuntu)

include_directories(${PROJECT_SOURCE_DIR}/include/opencv)
include_directories(${PROJECT_SOURCE_DIR}/include/tflite)

add_executable(face_recognition main.cpp)

add_library(lib_core SHARED IMPORTED)
set_target_properties(lib_core PROPERTIES IMPORTED_LOCATION ${LIB_DIR}/libopencv_core.so)

add_library(lib_highgui SHARED IMPORTED)
set_target_properties(lib_highgui PROPERTIES IMPORTED_LOCATION ${LIB_DIR}/libopencv_highgui.so)

add_library(lib_imgcodecs SHARED IMPORTED)
set_target_properties(lib_imgcodecs PROPERTIES IMPORTED_LOCATION ${LIB_DIR}/libopencv_imgcodecs.so)

add_library(lib_imgproc SHARED IMPORTED)
set_target_properties(lib_imgproc PROPERTIES IMPORTED_LOCATION ${LIB_DIR}/libopencv_imgproc.so)

add_library(lib_calib3d SHARED IMPORTED)
set_target_properties(lib_calib3d PROPERTIES IMPORTED_LOCATION ${LIB_DIR}/libopencv_calib3d.so)

add_library(lib_features2d SHARED IMPORTED)
set_target_properties(lib_features2d PROPERTIES IMPORTED_LOCATION ${LIB_DIR}/libopencv_features2d.so)

add_library(lib_flann SHARED IMPORTED)
set_target_properties(lib_flann PROPERTIES IMPORTED_LOCATION ${LIB_DIR}/libopencv_flann.so)

add_library(lib_videoio SHARED IMPORTED)
set_target_properties(lib_videoio PROPERTIES IMPORTED_LOCATION ${LIB_DIR}/libopencv_videoio.so)

add_library(lib_tflite SHARED IMPORTED)
set_target_properties(lib_tflite PROPERTIES IMPORTED_LOCATION ${LIB_DIR}/libtensorflowlite_c.so)

target_link_libraries(
    face_recognition PUBLIC 
    lib_core lib_highgui lib_imgcodecs lib_imgproc lib_calib3d lib_videoio lib_features2d lib_flann lib_tflite
)

# target_include_directories(face_recognition PUBLIC
#     "${PROJECT_BINARY_DIR}"
#     "${PROJECT_SOURCE_DIR}/opencv/include"
# )





