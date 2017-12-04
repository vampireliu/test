# cmake needs this line
cmake_minimum_required(VERSION 2.8)
#set(opencv_DIR"/home/liuxf/opencv-2.4.11/build")

# Define project name
project(tld_tracker)


# Find OpenCV, you may need to set OpenCV_DIR variable
# to the absolute path to the directory containing OpenCVConfig.cmake file
# via the command line or GUI
find_package(OpenCV REQUIRED)

# If the package has been found, several variables will
# be set, you can find the full list with descriptions
# in the OpenCVConfig.cmake file.
# Print some message showing some of them
message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${Open_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

#Set the cooridante src files
set(SRC_LIST FerNNClassifier.cpp LKTracker.cpp run_tld.cpp TLD.cpp tld_utils.cpp)

# Add OpenCV headers location to your include paths
include_directories(${OpenCV_INCLUDE_DIRS})

# Declare the executable target built from your sources
add_executable(run_tld ${SRC_LIST})

# Link your application with OpenCV libraries
target_link_libraries(run_tld ${OpenCV_LIBS})



