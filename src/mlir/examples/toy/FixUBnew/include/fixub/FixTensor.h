#ifndef FIX_TENSOR_H
#define FIX_TENSOR_H

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"

struct FixTensor {
  static void fixtensor(mlir::Operation *op);

  static void fixEmpty(mlir::Operation *op);

  static void fixDim(mlir::Operation *op);
  static void fixCast(mlir::Operation *op);

  static mlir::Operation *
  fixScatterUniqueIndices(mlir::tensor::ScatterOp scatterOp, mlir::Location loc,
                          mlir::OpBuilder builder);
  static mlir::Operation *traverseScatterIndices(
      mlir::tensor::ScatterOp scatterOp, int64_t curIndicesDim, int64_t rank,
      llvm::SmallVector<mlir::Value> indicesIdxs, mlir::ValueRange iterArgs,
      mlir::Location loc, mlir::OpBuilder builder);
  static mlir::Operation *
  findNotUseVal(int64_t curIndicesDim, int64_t rank,
                llvm::SmallVector<mlir::Value> indicesIdxs,
                mlir::ValueRange iterArgs, mlir::Location loc,
                mlir::OpBuilder builder);

  static void fixScatter(mlir::Operation *op);
  static void fixUnpack(mlir::Operation *op);
  static void fixPack(mlir::Operation *op);
  static void fixExtract(mlir::Operation *op);
  static void fixInsert(mlir::Operation *op);
};

#endif // FIX_TENSOR_H
