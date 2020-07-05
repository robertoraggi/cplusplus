project(pgen)

cmake_minimum_required(VERSION 3.9)

set(CMAKE_CXX_STANDARD 17)

aux_source_directory(. SOURCES)
add_executable(kwgen ${SOURCES})

install(TARGETS kwgen EXPORT kwgen DESTINATION bin)
