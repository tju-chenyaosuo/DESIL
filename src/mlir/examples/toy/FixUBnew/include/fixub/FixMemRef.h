#ifndef FIX_MEMREF_H
#define FIX_MEMREF_H

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"

struct FixMemref {
  static mlir::ModuleOp getParentModuleOp(mlir::Operation *op);
  static void fullMemrefWithRandVal(mlir::OpBuilder builder, mlir::Location loc,
                                    mlir::Value memref, mlir::Value lowerBound,
                                    mlir::Value upperBound);

  static void fixmemref(mlir::Operation *op);
  static void fixAssumeAlignment(mlir::Operation *op);
  static void fixCopy(mlir::Operation *op);
  static void fixRealloc(mlir::Operation *op);
  static void fixAlloc(mlir::Operation *op);
  static void fixCast(mlir::Operation *op);
  static void fixStore(mlir::Operation *op);
  static void fixLoad(mlir::Operation *op);
  static void fixDim(mlir::Operation *op);
};

#endif // FIX_MEMREF_H
