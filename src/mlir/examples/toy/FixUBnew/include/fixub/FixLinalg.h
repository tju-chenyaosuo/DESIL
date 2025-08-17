#ifndef FIX_LINALG_H
#define FIX_LINALG_H

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"

struct FixLinalg {
  static void fixlinalg(mlir::Operation *op);

  static void fixCopy(mlir::Operation *op);
  static void fixBroadcast(mlir::Operation *op);
  static void fixGeneric(mlir::Operation *op);
  static void fixMap(mlir::Operation *op);
  static void fixMatmul(mlir::Operation *op);
  static void fixTranspose(mlir::Operation *op);
};

#endif // FIX_LINALG_H
