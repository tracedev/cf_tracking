string(REGEX REPLACE "cmake$" "" cf_tracking_path ${CMAKE_CURRENT_LIST_FILE})

if (NOT TARGET cf_tracking)
  add_subdirectory(${cf_tracking_path} cf_tracking_build)
endif()
