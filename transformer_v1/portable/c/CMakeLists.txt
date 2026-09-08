cmake_minimum_required(VERSION 3.20)
project(coop_tracker_c C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
if(NOT ONNXRUNTIME_ROOT)
    message(FATAL_ERROR "Set ONNXRUNTIME_ROOT to the extracted ONNX Runtime C package")
endif()

add_library(coop_tracker STATIC src/tracker.c)
target_include_directories(coop_tracker PUBLIC include "${ONNXRUNTIME_ROOT}/include")
target_link_directories(coop_tracker PUBLIC "${ONNXRUNTIME_ROOT}/lib")
target_link_libraries(coop_tracker PUBLIC onnxruntime)

add_executable(verify_golden_case src/verify_golden_case.c)
target_link_libraries(verify_golden_case PRIVATE coop_tracker)
if(UNIX)
    target_link_libraries(verify_golden_case PRIVATE m)
endif()

if(WIN32)
    add_custom_command(TARGET verify_golden_case POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${ONNXRUNTIME_ROOT}/lib/onnxruntime.dll"
            "$<TARGET_FILE_DIR:verify_golden_case>")
endif()

if(MSVC)
    target_compile_options(coop_tracker PRIVATE /W4)
else()
    target_compile_options(coop_tracker PRIVATE -Wall -Wextra -Wpedantic)
endif()
