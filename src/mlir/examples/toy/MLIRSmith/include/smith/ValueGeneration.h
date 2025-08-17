//
// Created by Stan Wang on 2022/12/4.
//

#ifndef MLIR_FUZZ_VALUEGENERATION_H
#define MLIR_FUZZ_VALUEGENERATION_H

#include "Util.h"
#include "smith/TypeValue.h"


// helper to fill a memref
void fillMemRef(mlir::Location loc, mlir::OpBuilder &b, int dimension, mlir::Value alloc,
                mlir::MemRefType memType, std::map<int, mlir::Value> &constantIndices,
                llvm::SmallVector<mlir::Value> indices);

mlir::Value initVector(mlir::OpBuilder &builder, mlir::Location loc, mlir::VectorType type);

mlir::Value initMemref(mlir::OpBuilder &builder, mlir::Location loc, mlir::MemRefType type);



#endif // MLIR_FUZZ_VALUEGENERATION_H
