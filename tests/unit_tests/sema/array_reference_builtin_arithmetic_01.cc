// RUN: %cxx -std=c++26 -fsyntax-only -fvalidate-ast %s

using Bytes = unsigned char[16];
struct Data {
  Bytes bytes;
  operator Bytes&() { return bytes; }
  operator const Bytes&() const { return bytes; }
};
unsigned char* add(Data& data) { return data + 1; }
const unsigned char* add(const Data& data) { return data + 1; }
