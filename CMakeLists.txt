cmake_minimum_required(VERSION 3.0.0)
project(machine VERSION 0.1.0)

include(CTest)
enable_testing()

add_library(linalg "libs/linalg/linalg.cpp")
add_library(learn "libs/learn/learn.cpp")
add_library(image "libs/image/image.cpp")
add_library(data_loader "libs/data_loader/data_loader.cpp")

add_executable(${PROJECT_NAME} main.cpp)

target_link_libraries(learn linalg pthread)
target_link_libraries(image learn)
target_link_libraries(${PROJECT_NAME} linalg learn data_loader image pthread)

target_compile_options(linalg PRIVATE -Wall -Wextra -pedantic)
target_compile_options(learn PRIVATE -Wall -Wextra -pedantic)
target_compile_options(image PRIVATE -Wall -Wextra -pedantic)
target_compile_options(data_loader PRIVATE -Wall -Wextra -pedantic)
target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -pedantic)

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Ofast")
set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
