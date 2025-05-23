cmake_minimum_required(VERSION 3.20) # old CMake version
project(QuantumProject) # name of the executable of the project

# cmake_policy(SET CMP0167 NEW)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


# ======== Dependencies ========
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/external/spectra/include)

configure_file(${PROJECT_SOURCE_DIR}/plot.py ${PROJECT_BINARY_DIR} @ONLY)
configure_file(${PROJECT_SOURCE_DIR}/plot_mean_field.py ${PROJECT_BINARY_DIR} @ONLY)


# ======== Build ========
file(GLOB_RECURSE SRC_FILES src/*.cpp)
add_executable(${PROJECT_NAME} ${SRC_FILES})


# ======== Optimization ========
# Enable compiler optimizations
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")
endif()


# ======== Disable All Warnings ========
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(${PROJECT_NAME} PRIVATE -w)
    # Additional flags to suppress specific Clang warnings
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        target_compile_options(${PROJECT_NAME} PRIVATE
            -Wno-analyzer-too-complex
            -Wno-analyzer-infinite-loop
            -Wno-analyzer-null-dereference
            -Wno-analyzer-deadcode
            -Wno-analyzer-undefined-retval
            -Wno-analyzer-undefined-behavior
        )
    endif()
elseif(MSVC)
    target_compile_options(${PROJECT_NAME} PRIVATE /W0)
endif()


# ======== OpenMP ========
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    target_link_libraries(QuantumProject PRIVATE OpenMP::OpenMP_CXX)
    message(STATUS "Found OpenMP, enabling parallelization")
else()
    message(WARNING "OpenMP not found, parallelization disabled")
endif()
