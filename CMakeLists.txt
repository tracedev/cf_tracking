project(CfTracking)

cmake_minimum_required(VERSION 2.8)

# option to enable OpenMP; only relevant for the KCF version with the
# VOT scale estimation
option(WITH_OPENMP "Enable OpenMP" OFF)

if(WITH_OPENMP)
    find_package(OpenMP REQUIRED)

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif(WITH_OPENMP)

# add OpenCV
find_package(OpenCV REQUIRED highgui video videoio imgproc imgcodecs)

# add 3rdparty source
set(CF_CV_EXT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src/3rdparty/cv_ext")
set(CF_PIOTR_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src/3rdparty/piotr")

set(CF_TRACKING_INCLUDE "
  ${CF_CV_EXT_DIR};
  ${CF_PIOTR_DIR};
  ${CF_PIOTR_DIR}/src;
  ${CMAKE_CURRENT_SOURCE_DIR}/src/main;
  ${CMAKE_CURRENT_SOURCE_DIR}/src/cf_libs/kcf;
  ${CMAKE_CURRENT_SOURCE_DIR}/src/cf_libs/dsst;
  ${CMAKE_CURRENT_SOURCE_DIR}/src/cf_libs/common")
set(CF_TRACKING_INCLUDE ${CF_TRACKING_INCLUDE} PARENT_SCOPE)
include_directories(${CF_TRACKING_INCLUDE} ${OpenCV_INCLUDE_DIR})

set(CF_PIOTR_SOURCES
    ${CF_PIOTR_DIR}/src/gradientMex.cpp)

set(CF_LIB_COMMON_SOURCES
    src/cf_libs/common/math_helper.cpp
    src/cf_libs/common/scale_estimator.cpp
    ${CF_CV_EXT_DIR}/shift.cpp
    ${CF_CV_EXT_DIR}/math_spectrums.cpp
    ${CF_PIOTR_SOURCES})

set(CF_MAIN_SOURCES
    src/cf_libs/dsst/dsst_tracker.cpp
    src/cf_libs/kcf/kcf_tracker.cpp)

add_library(cf_tracking_lib STATIC
    ${CF_MAIN_SOURCES}
    ${CF_LIB_COMMON_SOURCES})

target_link_libraries(cf_tracking_lib ${OpenCV_LIBS})
