#ifndef FIX_VECTOR_H
#define FIX_VECTOR_H

#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"

struct FixVector {
  // Helper functions
  static void fixIndices(mlir::Operation *op);
  static mlir::Operation *getLastDimSize(mlir::Value memref);

  // Fix functions
  static void fixvector(mlir::Operation *op);
  static void fixMaskLoadAndStore(mlir::Operation *op);
  static void fixCompressStoreAndExpandLoad(mlir::Operation *op);
  static void fixgather(mlir::Operation *op);
  static void fixscatter(mlir::Operation *op);
  static void fixInsertElement(mlir::Operation *op);
  static void fixLoad(mlir::Operation *op);
};

#endif // FIX_VECTOR_H
