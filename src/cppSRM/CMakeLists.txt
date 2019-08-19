cmake_minimum_required(VERSION 2.8.9)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++1z -Wall")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/modules/")

set(CMAKE_MACOSX_RPATH 1)
project (SRM C CXX)


set(CMAKE_CXX_STANDARD 17) 

find_package( ZLIB REQUIRED )

message(STATUS "Checking OpenMP")
find_package(OpenMP REQUIRED)
IF(OPENMP_FOUND)
    message("Found OpenMP!")
    # add flags for OpenMP
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${OpenMP_SHARED_LINKER_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
ELSE()
    message("Missed OpenMP!")
ENDIF()

MESSAGE(WARNING ${OpenMP_C_FLAGS})
MESSAGE(WARNING ${OpenMP_CXX_FLAGS})

#SET( OpenMP_CXX_FLAGS "-fopenmp" )

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    # using Clang
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    # using GCC
    SET( CMAKE_OPENMP_LIBRARY_NAME "omp")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
    # using Intel C++
    SET( CMAKE_OPENMP_LIBRARY_NAME "")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    # using Visual Studio C++
endif()

file(GLOB lib_H
        "src/*.h"
)

file(GLOB lib_SRC
        "src/*.cpp"
)


include_directories(${CMAKE_SOURCE_DIR})

set (SOURCE segment.cpp )
set( CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib )

add_library(SRM MODULE ${SOURCE} ${lib_SRC} ${lib_H})

TARGET_LINK_LIBRARIES(SRM z ${CMAKE_OPENMP_LIBRARY_NAME})

