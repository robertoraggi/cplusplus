# Copyright (c) 2026 Roberto Raggi <roberto.raggi@gmail.com>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER

add_library(cxx-link
    cxx/linker/archive.cc
    cxx/linker/bytes.cc
    cxx/linker/driver.cc
    cxx/linker/layout.cc
    cxx/linker/linker.cc
    cxx/linker/object.cc
    cxx/linker/relocate.cc
    cxx/linker/symbols.cc
    cxx/linker/writer.cc
)

target_include_directories(cxx-link PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(cxx-link PUBLIC cxx_std_23)
