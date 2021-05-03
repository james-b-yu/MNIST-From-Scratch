cmake_minimum_required(VERSION 3.0.0)
project(machine VERSION 0.1.0)

include(CTest)
enable_testing()

add_library(learn "libs/learn/learn.cpp")
add_library(image "libs/image/image.cpp")
add_library(character "libs/image/character.cpp")
add_library(linear "libs/image/linear.cpp")
add_library(image_reversed "libs/image/image_reversed.cpp")
add_library(data_loader "libs/data_loader/data_loader.cpp")

add_executable(${PROJECT_NAME} main.cpp)

target_link_libraries(learn pthread boost_iostreams boost_serialization)
target_link_libraries(image learn)
target_link_libraries(character learn)
target_link_libraries(linear learn)
target_link_libraries(image_reversed learn)
target_link_libraries(${PROJECT_NAME} learn data_loader image image_reversed character linear pthread)

target_compile_options(learn PRIVATE -Wall -Wextra -pedantic)
target_compile_options(image PRIVATE -Wall -Wextra -pedantic)
target_compile_options(data_loader PRIVATE -Wall -Wextra -pedantic)
target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -pedantic)

set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
