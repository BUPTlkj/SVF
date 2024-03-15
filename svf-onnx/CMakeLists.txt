find_package(Python 3.11 EXACT COMPONENTS Interpreter Development REQUIRED)

#target_link_libraries (${PROJECT_NAME} ${PYTHON_DL})

add_executable(svf-onnx
        SVFONNX.cpp
        SVFONNX.h
        Solver.cpp
        Solver.h
        Solver.h
        CheckModels.cpp
        CheckModels.h
        CheckModels.h
        IntervalSolver.cpp
        IntervalSolver.h)

message("Python INCLUDE DIR: ${Python_INCLUDE_DIRS}")
message("Python LIB DIR: ${Python_LIBRARIES}")
message("SVF BUILD INCLUDE DIR: ${CMAKE_SOURCE_DIR}/svf/include")
message("SVF INSTALL INCLUDE DIR: ${SVF_INSTALL_INCLUDE_DIR}")
# Add the public headers as an include directory

#target_include_directories(svf-onnx PRIVATE ${Python_INCLUDE_DIRS})
target_include_directories(svf-onnx PRIVATE $<BUILD_INTERFACE: ${CMAKE_SOURCE_DIR}/svf/include>
        $<INSTALL_INTERFACE:${SVF_INSTALL_INCLUDE_DIR} ${Python_INCLUDE_DIRS})
target_link_libraries(svf-onnx PRIVATE ${Python_LIBRARIES} SvfCore)