add_llvm_executable(ae ae.cpp)

find_package(PkgConfig REQUIRED)
pkg_check_modules(PCRE REQUIRED libpcre)


include_directories(${PCRE_INCLUDE_DIRS})
link_directories(${PCRE_LIBRARY_DIRS})

target_link_libraries(ae PUBLIC ${llvm_libs} SvfLLVM ${Python_LIBRARIES} SvfONNX ${PCRE_LIBRARIES})
target_include_directories(ae PRIVATE ${Python_INCLUDE_DIRS} ${PCRE_INCLUDE_DIRS})
