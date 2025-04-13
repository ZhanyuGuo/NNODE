cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(grad_demo)

find_package(Torch REQUIRED)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(grad_demo grad_demo.cpp)
target_link_libraries(grad_demo "${TORCH_LIBRARIES}")
set_property(TARGET grad_demo PROPERTY CXX_STANDARD 17)
