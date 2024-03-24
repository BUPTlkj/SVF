add_llvm_executable(ae ae.cpp)

target_link_libraries(ae PUBLIC ${llvm_libs} SvfLLVM ${Python_LIBRARIES} SvfONNX)
target_include_directories(ae PRIVATE ${Python_INCLUDE_DIRS})
