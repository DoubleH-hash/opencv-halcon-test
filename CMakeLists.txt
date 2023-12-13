cmake_minimum_required(VERSION 3.10)

add_compile_options("$<$<NOT:$<CONFIG:Debug>>:/Zi>")
add_link_options("$<$<NOT:$<CONFIG:Debug>>:/DEBUG>")
add_link_options("$<$<NOT:$<CONFIG:Debug>>:/OPT:REF>")
add_link_options("$<$<NOT:$<CONFIG:Debug>>:/OPT:ICF>")

project(OpenCV)
set(CMAKE_CXX_STANDARD_REQUIRED on)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(Torch_DIR "D:\\OpenCV_test\\OpenCV\\include\\torch\\share\\cmake\\Torch")

set(OpenCV_DIR "D:\\OpenCV_test\\OpenCV\\include\\openv_cuda\\x64\\vc16\\lib")
message(STATUS "OPENCV:${CMAKE_SOURCE_DIR}")
message(STATUS "OPENCV:${OpenCV_DIR}")

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/include/Classes)
include_directories(${PROJECT_SOURCE_DIR}/include/Include)
include_directories("D:\\OpenCV_test\\OpenCV\\include")

#MVS header file include
include_directories((${PROJECT_SOURCE_DIR}/include/spdlog))

include_directories("D:\\OpenCV_test\\OpenCV\\include\\halcon")
include_directories("D:\\OpenCV_test\\OpenCV\\include\\halcon\\halconcpp")
include_directories("D:\\OpenCV_test\\OpenCV\\include\\halcon\\hdevengine")



find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)
message("opencv_lib:${OpenCV}")
include_directories(include ${OpenCV_INCLUDE_DIRS})
message("opencv_lib:${OpenCV_INCLUDE_DIRS}")

link_directories(${PROJECT_SOURCE_DIR}/libs/)
#MVS lib 
link_directories(${PROJECT_SOURCE_DIR}/libs/win64/)

link_directories("D:\\OpenCV_test\\OpenCV\\lib\\halcon")



add_executable(${CMAKE_PROJECT_NAME} "src/main.cpp"  "src/halconAction.cpp"  "src/Yolo.cpp" "src/Yolo.h"
									"src/halconAction.h" "src/opencvAction.cpp"  "src/opencvAction.h"    "src/CommonContent.h"   )

#add_executable(${CMAKE_PROJECT_NAME} "src/main.cpp"  "src/Yolo.cpp" "src/Yolo.h" "src/opencvAction.cpp"  "src/opencvAction.h")

target_link_libraries(${CMAKE_PROJECT_NAME} easynirlibd)

target_link_libraries(${CMAKE_PROJECT_NAME} "${TORCH_LIBRARIES}" "${OpenCV_LIBS}")

target_link_libraries(${CMAKE_PROJECT_NAME} Halcon)

target_link_libraries(${CMAKE_PROJECT_NAME} halconcpp)

target_link_libraries(${CMAKE_PROJECT_NAME} HDevEngineCpp)

message("OPENCV: ${OpenCV_LIBS}")