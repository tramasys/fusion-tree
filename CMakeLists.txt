cmake_minimum_required(VERSION 3.25)

project(fusion_tree VERSION 0.1.0 LANGUAGES CXX)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

add_library(fusion_tree INTERFACE)
add_library(fusion_tree::fusion_tree ALIAS fusion_tree)
target_compile_features(fusion_tree INTERFACE cxx_std_23)
target_include_directories(
  fusion_tree
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

if(PROJECT_IS_TOP_LEVEL)
  option(FUSION_TREE_BUILD_TESTS "Build the test suite" ON)
  option(FUSION_TREE_BUILD_BENCHMARKS "Build the benchmark" OFF)
  option(FUSION_TREE_ENABLE_SANITIZERS "Enable ASan and UBSan" OFF)
else()
  option(FUSION_TREE_BUILD_TESTS "Build the test suite" OFF)
  option(FUSION_TREE_BUILD_BENCHMARKS "Build the benchmark" OFF)
endif()

function(fusion_tree_set_project_options target)
  if(MSVC)
    target_compile_options(${target} PRIVATE /W4 /permissive-)
  else()
    target_compile_options(
      ${target}
      PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
  endif()

  if(FUSION_TREE_ENABLE_SANITIZERS AND NOT MSVC)
    target_compile_options(${target} PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
    target_link_options(${target} PRIVATE -fsanitize=address,undefined)
  endif()
endfunction()

if(FUSION_TREE_BUILD_TESTS)
  include(CTest)
  add_executable(fusion_tree_tests tests/fusion_set_test.cpp)
  target_link_libraries(fusion_tree_tests PRIVATE fusion_tree::fusion_tree)
  fusion_tree_set_project_options(fusion_tree_tests)
  add_test(NAME fusion_tree_tests COMMAND fusion_tree_tests)
endif()

if(FUSION_TREE_BUILD_BENCHMARKS)
  add_executable(fusion_tree_benchmark benchmarks/lookup_benchmark.cpp)
  target_link_libraries(fusion_tree_benchmark PRIVATE fusion_tree::fusion_tree)
  fusion_tree_set_project_options(fusion_tree_benchmark)
endif()

install(TARGETS fusion_tree EXPORT fusion_treeTargets)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(
  EXPORT fusion_treeTargets
  FILE fusion_treeTargets.cmake
  NAMESPACE fusion_tree::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/fusion_tree)

configure_package_config_file(
  cmake/fusion_treeConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/fusion_treeConfig.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/fusion_tree)
write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/fusion_treeConfigVersion.cmake
  COMPATIBILITY SameMajorVersion)
install(
  FILES
    ${CMAKE_CURRENT_BINARY_DIR}/fusion_treeConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/fusion_treeConfigVersion.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/fusion_tree)

