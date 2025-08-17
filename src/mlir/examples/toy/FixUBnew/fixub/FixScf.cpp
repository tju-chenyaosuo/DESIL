#include <cmath>
#include <random>

#include "mlir/InitAllDialects.h"

#include "fixub/ConstantPool.h"
#include "fixub/FixScf.h"
#include "fixub/FixUBUtils.h"

void FixScf::fixscf(mlir::Operation *op) {
  std::string opname = op->getName().getStringRef().str();
  if (opname == "scf.condition") {
    FixScf::fixcondition(op);
  }
  return;
}

void FixScf::fixcondition(mlir::Operation *op) {
  /**
   * This function is not for fixing the OOM problem when facing true while
   * loop.
   */
  mlir::OpBuilder builder(op->getBlock(), op->getBlock()->begin());
  builder.setInsertionPoint(op);
  mlir::Location loc = op->getLoc();

  op->setOperand(0, ConstantPool::getValue(0, builder.getI1Type()));
}