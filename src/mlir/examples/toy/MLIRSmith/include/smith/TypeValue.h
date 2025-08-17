//
// Created by Stan Wang on 2022/11/14.
//

#ifndef MLIR_FUZZ_TYPEVALUE_H
#define MLIR_FUZZ_TYPEVALUE_H

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/LLVMIR/LLVMOps.h.inc"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "toy/Dialect.h"
#include "toy/Passes.h"
#include "llvm/ADT/Sequence.h"

class TypeValue {
public:
  mlir::Type type;
  mlir::Value val;

  bool operator==(const TypeValue tval) const {
    return this->type == tval.type && this->val == tval.val;
  }

  TypeValue(mlir::Type type, mlir::Value val) : type(val.getType()), val(val) {
    assert(val);
  }
};

using TypeValueFilter = std::function<bool(TypeValue)>;

inline TypeValueFilter typeEquivalentFilter(mlir::Type &type) {
  return [&](TypeValue tval) { return tval.type == type; };
}

inline TypeValueFilter emptyFilter() {
  return [&](TypeValue tval) { return true; };
}

#endif // MLIR_FUZZ_TYPEVALUE_H
