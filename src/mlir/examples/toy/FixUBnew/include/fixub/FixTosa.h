#ifndef FIX_TOSA_H
#define FIX_TOSA_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"

struct FixTosa {
  static void fixtosa(mlir::Operation *op);

  static mlir::RankedTensorType
  genStaticTensorTyFromExisting(mlir::ShapedType sty, mlir::Type elemTy,
                                mlir::OpBuilder builder);
  static mlir::Type randomIntegerType(mlir::OpBuilder builder);
  static mlir::Type randomFloatType(mlir::OpBuilder builder);
  static void FixTranspose_conv2d(mlir::Operation *op);
};

#endif // FIX_TOSA_H
