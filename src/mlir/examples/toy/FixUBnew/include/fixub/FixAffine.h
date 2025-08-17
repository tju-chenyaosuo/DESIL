#ifndef FIX_AFFINE_H
#define FIX_AFFINE_H

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"

struct FixAffine {
  static void fixaffine(mlir::Operation *op);

  // static mlir::Value genValidIndex(int64_t dimIdx, mlir::Value dimVal);
  static bool isValidAffineIndexOperand(mlir::Value value, mlir::Region *region);

  static void fixStoreLoad(mlir::Operation *op);
  static void fixVectorStoreLoad(mlir::Operation *op);
  static void fixYield(mlir::Operation *op);
};

#endif // FIX_AFFINE_H
