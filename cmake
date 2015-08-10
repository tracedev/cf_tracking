cmake_minimum_required(VERSION 2.6)

string(REGEX REPLACE "cmake$" "" cf_tracking_path ${CMAKE_CURRENT_LIST_FILE})

if (NOT TARGET cf_tracking)
  add_subdirectory(${cf_tracking_path} cf_tracking_build)
endif()

# Added by subdirectory
include_directories(${CF_INCLUDE})
