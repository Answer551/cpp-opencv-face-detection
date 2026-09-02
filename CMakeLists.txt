cmake_minimum_required(VERSION 3.20)

project(cpp_opencv_face_detection VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Pass this path at configure time; see README.md for the local command.
set(OpenCV_DIR "" CACHE PATH "Directory containing OpenCVConfig.cmake")
find_package(OpenCV 4.10 REQUIRED COMPONENTS core dnn highgui imgcodecs imgproc objdetect videoio)

add_executable(face_identity_tool src/main.cpp)
target_include_directories(face_identity_tool PRIVATE "${CMAKE_SOURCE_DIR}/third_party")
target_link_libraries(face_identity_tool PRIVATE ${OpenCV_LIBS})

if(WIN32)
    set(OPENCV_RUNTIME_DIR "${OpenCV_DIR}/bin" CACHE PATH
        "Directory containing OpenCV runtime DLLs")
    file(GLOB OPENCV_RUNTIME_DLLS "${OPENCV_RUNTIME_DIR}/libopencv_*.dll")
    if(NOT OPENCV_RUNTIME_DLLS)
        message(FATAL_ERROR
            "No OpenCV DLLs found. Set OPENCV_RUNTIME_DIR to the OpenCV bin directory.")
    endif()
    foreach(runtime_dll IN LISTS OPENCV_RUNTIME_DLLS)
        add_custom_command(TARGET face_identity_tool POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${runtime_dll}" "$<TARGET_FILE_DIR:face_identity_tool>")
    endforeach()
endif()
