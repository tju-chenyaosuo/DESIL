#include "mlir/InitAllDialects.h"
#include "mlir/Parser/Parser.h"

#include "fixub/ConstantPool.h"
#include "fixub/FixUBUtils.h"

std::string ConstantPool::getIden(int64_t v, mlir::Type t) {
  /**
   * Generate the constant identifier.
   * Format: value-type; Example: -111-index
   *
   * Params：
   * int_64t v, the constant value.
   * mlir::Type t, the type.
   *
   * Return:
   * std::string that represent a constant value of mlir::Value in ConstantPool.
   */
  return std::to_string(v) + FixUBUtils::getValueTypeStr(t);
}

mlir::Value ConstantPool::getValue(int64_t v, mlir::Type t) {
  /**
   * Get (or get after generate) the constant value with specific type.
   *
   * Params:
   * int_64t v, the constant value.
   * mlir::Type t, the required mlir::Type of mlir::Value.
   *
   * Return:
   * The constant mlir::Value maintained in ConstantPool.
   */

  std::string key = getIden(v, t);
  mlir::Value retVal;
  if (constantPool.find(key) != constantPool.end()) {
    retVal = constantPool[key];
  } else {
    // Turn to the front of the function.
    mlir::Operation *firstOpInFunc =
        &(ConstantPool::funcOp->getRegion(0).front().getOperations().front());
    mlir::Location loc = firstOpInFunc->getLoc();
    mlir::OpBuilder builder(firstOpInFunc->getBlock(),
                            firstOpInFunc->getBlock()->begin());
    builder.setInsertionPoint(firstOpInFunc);

    assert(t.isIntOrIndex());
    if (t.isIndex()) {
      retVal = builder
                   .create<mlir::index::ConstantOp>(
                       loc, mlir::IntegerAttr::get(t, v))
                   .getOperation()
                   ->getResult(0);
    } else {
      retVal = builder
                   .create<mlir::arith::ConstantOp>(
                       loc, mlir::IntegerAttr::get(t, v))
                   .getOperation()
                   ->getResult(0);
    }
    constantPool[key] = retVal;
  }
  assert(retVal);
  return retVal;
}