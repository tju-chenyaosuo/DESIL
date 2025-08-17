
#ifndef FIX_ARITH_H
#define FIX_ARITH_H

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"

struct FixArith {

  static void fixarith(mlir::Operation *op);

  static void fixshlr(mlir::Operation *op);
  static void fixdivsi(mlir::Operation *op);
  static void fixremsi(mlir::Operation *op);
  static void fixdivui(mlir::Operation *op);
};

#endif // FIX_ARITH_H
