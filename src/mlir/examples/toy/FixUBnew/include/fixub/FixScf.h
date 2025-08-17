
#ifndef FIX_SCF_H
#define FIX_SCF_H

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"

struct FixScf {
  static void fixscf(mlir::Operation *op);
  static void fixcondition(mlir::Operation *op);
};

#endif // FIX_SCF_H
