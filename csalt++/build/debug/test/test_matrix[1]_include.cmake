if(EXISTS "/home/jonwebb/Code_Dev/C++/csalt++/csalt++/build/debug/test/test_matrix[1]_tests.cmake")
  include("/home/jonwebb/Code_Dev/C++/csalt++/csalt++/build/debug/test/test_matrix[1]_tests.cmake")
else()
  add_test(test_matrix_NOT_BUILT test_matrix_NOT_BUILT)
endif()
