// RUN: %cxx -E -P %s -o - | %filecheck %s

// clang-format off

#define BOOST_PP_STRINGIZE(text) BOOST_PP_STRINGIZE_I(text)
#define BOOST_PP_STRINGIZE_I(...) #__VA_ARGS__

auto fn = BOOST_PP_STRINGIZE_I(boost/mpl/aux_/preprocessed/AUX778076_PREPROCESSED_HEADER);

// CHECK: auto fn = "boost/mpl/aux_/preprocessed/AUX778076_PREPROCESSED_HEADER";
