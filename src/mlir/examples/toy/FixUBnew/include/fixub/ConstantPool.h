#ifndef CONSTANT_POOL_H
#define CONSTANT_POOL_H

#include <map>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

struct ConstantPool {
  inline static mlir::Operation *funcOp = nullptr;
  inline static std::map<std::string, mlir::Value> constantPool = {};
  static std::string getIden(int64_t v, mlir::Type t);
  static mlir::Value getValue(int64_t v, mlir::Type t);
};

#endif // CONSTANT_POOL_H