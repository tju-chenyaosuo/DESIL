#ifndef FIX_INDEX_H
#define FIX_INDEX_H

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"

struct FixIndex {

  static void fixindex(mlir::Operation *op);

  static void fixshlr(mlir::Operation *op);
  static void fixdivsi(mlir::Operation *op);
  static void fixremsi(mlir::Operation *op);
  static void fixdivui(mlir::Operation *op);

  static const int INDEX_BITWIDTH = 64;
};

#endif // FIX_INDEX_H
