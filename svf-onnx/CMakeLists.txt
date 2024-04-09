#target_link_libraries (${PROJECT_NAME} ${PYTHON_DL})

add_library(SvfONNX SVFONNX.cpp
        NNLoaddata.cpp)

message("Python INCLUDE DIR: ${Python_INCLUDE_DIRS}")
message("Python LIB DIR: ${Python_LIBRARIES}")
message("SVF BUILD INCLUDE DIR: ${CMAKE_SOURCE_DIR}/svf/include")
message("SVF INSTALL INCLUDE DIR: ${SVF_INSTALL_INCLUDE_DIR}")
# Add the public headers as an include directory

target_include_directories(SvfONNX PRIVATE $<BUILD_INTERFACE: ${CMAKE_SOURCE_DIR}/svf/include>
        $<INSTALL_INTERFACE:${SVF_INSTALL_INCLUDE_DIR} ${Python_INCLUDE_DIRS})

target_link_libraries(SvfONNX PRIVATE ${Python_LIBRARIES} SvfCore)
