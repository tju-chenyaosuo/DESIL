# For a better template to copy, see examples/standalone
include_directories(include)
add_subdirectory(include)

set(LLVM_LINK_COMPONENTS
  Support
  )

set(LLVM_TARGET_DEFINITIONS mlir/ToyCombine.td)
mlir_tablegen(ToyCombine.inc -gen-rewriters)
add_public_tablegen_target(ToyCh5_dynamicLoweringCombineIncGen)

add_toy_chapter(MLIRSmith
  toyc.cpp
  parser/AST.cpp
  mlir/MLIRGen.cpp
  mlir/Dialect.cpp
  mlir/LowerToAffineLoops.cpp
  mlir/ShapeInferencePass.cpp
  mlir/ToyCombine.cpp
  smith/MLIRSmith.cpp
        smith/generators/OpGeneratorProvider.cpp
        smith/generators/ArithGeneratorProvider.cpp
        smith/generators/BufferizationGeneratorProvider.cpp
        smith/generators/IndexGeneratorProvider.cpp
        smith/generators/LinalgGeneratorProvider.cpp
        smith/generators/MathGeneratorProvider.cpp
        smith/generators/MemrefGeneratorProvider.cpp
        smith/generators/FuncGeneratorProvider.cpp
        smith/generators/SCFGeneratorProvider.cpp
        smith/generators/TensorGeneratorProvider.cpp
        smith/generators/VectorGeneratorProvider.cpp
        smith/generators/SPIRVGeneratorProvider.cpp
        smith/generators/SPIRVGeneratorProvider.cpp
        smith/generators/TosaGeneratorProvider.cpp
  smith/ValueGeneration.cpp
  smith/Util.cpp
  smith/RegionGeneration.cpp
  smith/TypedValuePool.cpp
  smith/TypeValue.cpp
        smith/generators/AffineGeneratorProvider.cpp
  DEPENDS
  ToyCh5_dynamicLoweringShapeInferenceInterfaceIncGen
  ToyCh5_dynamicLoweringOpsIncGen
  ToyCh5_dynamicLoweringCombineIncGen
  )

include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include/)
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)
target_link_libraries(MLIRSmith
  PRIVATE
    ${dialect_libs}
    ${extension_libs}
    MLIRAnalysis
    MLIRCallInterfaces
    MLIRCastInterfaces
    MLIRFunctionInterfaces
    MLIRIR
    MLIRParser
    MLIRPass
    MLIRSideEffectInterfaces
    MLIRSupport
    MLIRTransforms)
