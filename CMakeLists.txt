#
# Copyright 2018 TASS International.
#
# All rights reserved.
#
# Author: Changjie Guan (changjie.guan@tassinternational.com)
#
project(pso_cpu)

cmake_minimum_required (VERSION 3.5)


#Flags for compiler
IF(("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU") OR ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang"))
    set(CMAKE_CXX_FLAGS "-std=c++11 -O3")
ENDIF()


# Set output directory
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(LIB_NAME pso_lib)

include_directories(/usr/local/include
					/usr/local/cuda/samples/common/inc
)

link_directories(/usr/local/lib)


# OpenMP
OPTION (USE_OpenMP "Use OpenMP" ON)
IF(USE_OpenMP)
  FIND_PACKAGE(OpenMP)
  IF(OPENMP_FOUND)
    SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  ENDIF()
ENDIF()

set(SRC test_pso.cpp pso.cpp)

add_library(${LIB_NAME} pso.cpp)

add_executable(pso_test ${SRC})

target_link_libraries(pso_test ${LIB_NAME} python2.7)

