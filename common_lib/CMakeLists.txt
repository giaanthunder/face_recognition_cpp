cmake_minimum_required(VERSION 3.4.1)
set(CMAKE_CXX_STANDARD 11)

include_directories(../opencv/include)
add_library(lib_opencv SHARED IMPORTED)
set_target_properties(lib_opencv PROPERTIES IMPORTED_LOCATION ../jniLibs/x86_64/libopencv_java4.so)

include_directories(../tflite/headers)
add_library(lib_tflite SHARED IMPORTED)
set_target_properties(lib_tflite PROPERTIES IMPORTED_LOCATION ../jniLibs/x86_64/libtensorflowlite_c_delegate.so)

add_library(common_lib SHARED common_lib.cpp)
target_link_libraries(common_lib lib_opencv lib_tflite)
