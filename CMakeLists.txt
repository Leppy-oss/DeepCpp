cmake_minimum_required(VERSION 4.0)
project(cpp_torch)

include(CTest)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_compile_definitions(PROJECT_ROOT="${CMAKE_SOURCE_DIR}")
include_directories(include)
enable_testing()

include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.17.0
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
# set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE) 
FetchContent_MakeAvailable(googletest)

file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS "src/*.cpp")
list(FILTER SOURCES EXCLUDE REGEX ".*main\\.cpp$")

add_library(core ${SOURCES})

add_executable(cpp_torch src/main.cpp)
target_link_libraries(cpp_torch PRIVATE core)

file(GLOB_RECURSE TEST_SOURCES CONFIGURE_DEPENDS "tests/*.cpp")
add_executable(unit_tests ${TEST_SOURCES})

target_link_libraries(unit_tests PRIVATE core GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(unit_tests DISCOVERY_MODE PRE_TEST)
