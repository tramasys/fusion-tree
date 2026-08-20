cmake_minimum_required(VERSION 3.25)
project(fusion_tree_package_consumer LANGUAGES CXX)

find_package(fusion_tree CONFIG REQUIRED)
add_executable(package_consumer main.cpp)
target_link_libraries(package_consumer PRIVATE fusion_tree::fusion_tree)

