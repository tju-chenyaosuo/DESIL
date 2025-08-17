# For a better template to copy, see examples/standalone
include_directories(include)
add_subdirectory(include)

set(LLVM_LINK_COMPONENTS
  Support
  )


add_toy_chapter(FixUBnew
  FixUBMain.cpp
  fixub/FixAffine.cpp
  fixub/FixArith.cpp
  fixub/FixIndex.cpp
  fixub/FixLinalg.cpp
  fixub/FixTensor.cpp
  fixub/FixVector.cpp
  fixub/FixMemRef.cpp
  fixub/FixTosa.cpp
  fixub/FixUBUtils.cpp
  fixub/FixScf.cpp
  fixub/ConstantPool.cpp
  DEPENDS)

include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include/)
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
target_link_libraries(FixUBnew
  PRIVATE
    ${dialect_libs}
    MLIRAnalysis
    MLIRCallInterfaces
    MLIRCastInterfaces
    MLIRIR
    MLIRParser
    MLIRPass
    MLIRSideEffectInterfaces
    MLIRSupport
    MLIRTransforms)
