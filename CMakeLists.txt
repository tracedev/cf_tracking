# License Agreement (3-clause BSD License)
# Copyright (c) 2015, Klaus Haag, all rights reserved.
# Third party copyrights and patents are property of their respective owners.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the names of the copyright holders nor the names of the contributors
#   may be used to endorse or promote products derived from this software
#   without specific prior written permission.
#
# This software is provided by the copyright holders and contributors "as is" and
# any express or implied warranties, including, but not limited to, the implied
# warranties of merchantability and fitness for a particular purpose are disclaimed.
# In no event shall copyright holders or contributors be liable for any direct,
# indirect, incidental, special, exemplary, or consequential damages
# (including, but not limited to, procurement of substitute goods or services;
# loss of use, data, or profits; or business interruption) however caused
# and on any theory of liability, whether in contract, strict liability,
# or tort (including negligence or otherwise) arising in any way out of
# the use of this software, even if advised of the possibility of such damage.

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
    ${CF_CV_EXT_DIR}/shift.cpp
    ${CF_CV_EXT_DIR}/math_spectrums.cpp
    ${CF_PIOTR_SOURCES})

set(CF_MAIN_SOURCES
    src/main/image_acquisition.cpp)

add_library(cf_tracking_lib STATIC
    ${CF_MAIN_SOURCES}
    ${CF_LIB_COMMON_SOURCES})

target_link_libraries(cf_tracking_lib ${OpenCV_LIBS})
