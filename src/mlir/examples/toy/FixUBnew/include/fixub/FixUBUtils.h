#ifndef FIX_UB_UTILS_H
#define FIX_UB_UTILS_H

#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"

struct FixUBUtils {
  static mlir::IntegerAttr getI(mlir::Value val);
  static std::int64_t randsval(int64_t bitwidth);
  static std::uint64_t randuval(uint64_t bitwidth);
  static std::int64_t bitsmax(int64_t bitwidth);
  static std::uint64_t bitumax(uint64_t bitwidth);
  static std::int64_t bitsmin(int64_t bitwidth);
  static std::int64_t randsval(int64_t lb, int64_t ub);
  static std::string getOperationNameParamType(mlir::Operation *o);
  static std::string getValueTypeStr(mlir::Value v);
  static std::string getValueTypeStr(mlir::Type vTy);
  static std::string getLocationStr(mlir::Operation *o);
  static std::int64_t getLineNumber(mlir::Operation *o);
  static bool isDynShape(mlir::ShapedType sty);
  static mlir::Value genRandomTensorFromExisting(mlir::ShapedType tensorTy,
                                                 mlir::OpBuilder builder,
                                                 mlir::Location loc);
  static mlir::Value genRandomIntOrFloatValue(mlir::OpBuilder builder,
                                              mlir::Location loc,
                                              mlir::Type type);

  static void getShapes(mlir::OpBuilder builder, mlir::Location loc,
                        mlir::Value val,
                        llvm::SmallVector<mlir::Value> *shapes);
  static void getDynDims(mlir::OpBuilder builder, mlir::Location loc,
                         mlir::Value val,
                         llvm::SmallVector<mlir::Value> *shapes,
                         llvm::SmallVector<int64_t> *idx);
  static mlir::Operation *compareShape(mlir::OpBuilder builder,
                                       mlir::Location loc,
                                       llvm::SmallVector<mlir::Value> shape1,
                                       llvm::SmallVector<mlir::Value> shape2);
  static mlir::Operation *castToAllDyn(mlir::OpBuilder builder,
                                       mlir::Location loc, mlir::Value val);
  static void getValidIndex(mlir::OpBuilder builder, mlir::Location loc,
                            llvm::SmallVector<mlir::Value> dims,
                            llvm::SmallVector<mlir::Value> oldShape,
                            llvm::SmallVector<mlir::Value> *newShape);
  static void getDynDims(mlir::OpBuilder builder, mlir::Location loc,
                         mlir::Value val,
                         llvm::SmallVector<mlir::Value> *shapes);
  static mlir::Operation* getDifiningOpIncludeArgs(mlir::OpBuilder builder, mlir::Location loc, mlir::Value val);
};

#endif // FIX_UB_UTILS_H
