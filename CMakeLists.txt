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
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -Wno-deprecated-declarations")
endif()


# ======== OpenMP ========
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    target_link_libraries(QuantumProject PRIVATE OpenMP::OpenMP_CXX)
    message(STATUS "Found OpenMP, enabling parallelization")
else()
    message(WARNING "OpenMP not found, parallelization disabled")
endif()


# ======== Doxygen ========
#find_package(Doxygen)
#if (DOXYGEN_FOUND)
#set(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile)
#set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)
#configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)
#add_custom_target(doc_doxygen
#COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
#WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
#COMMENT "Generating API documentation with Doxygen"
# VERBATIM
#)
#else (DOXYGEN_FOUND)
    #message(STATUS "Doxygen not found, unable to generate documentation")
#endif (DOXYGEN_FOUND)


# ========== Qt =========
#find_package(Qt5 COMPONENTS Core Gui Widgets REQUIRED)
#qt_standard_project_setup()
#qt_add_executable(project main.cpp)
#target_link_libraries(${PROJECT_NAME} Qt5::Core Qt5::Gui Qt5::Widgets)



# Add compiler warning flags
target_compile_options(${PROJECT_NAME} PRIVATE -march=native)


