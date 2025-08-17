//
// Created by Stan Wang on 2022/10/31.
//

#ifndef MLIR_FUZZ_TYPEGENERATION_H
#define MLIR_FUZZ_TYPEGENERATION_H

#include "Util.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"

// using namespace mlir;

// TODO- Migrate to Conf
const int rank_ub = 3;
const int dim_ub = 32;
const int idx_ub = dim_ub;
const int func_arg_ub = 4;
const int func_ret_ub = 2; // the fun either have one return value or none.
const int dimPoolSize = 4;
const int shapePoolSize = 4;

inline std::vector<int64_t> dimPool;
inline std::vector<std::vector<int64_t>> shapePool; // non-zero-ranked Shapes

inline long long random(long long mod) {
  long long rd = ((long long)rand() << 32) | rand();
  return rd % mod;
}

inline int randomRank() { return random(rank_ub) + 1; }

inline void initType() {
  for (int i = 0; i < dimPoolSize; i++) {
    dimPool.push_back(random(dim_ub) + 1);
  }
  // TODO-support dynamic shape
  // dimPool.push_back(ShapedType::kDynamicSize);

  // init shape pool
  for (int i = 0; i < shapePoolSize; i++) {
    std::vector<int64_t> dims;
    int sizesum = 1;
    for (int j = 0; j < randomRank(); j++) {
      int randomdim = random(dimPoolSize);
      if (sizesum * dimPool[randomdim] <= 500) {
        dims.push_back(dimPool[randomdim]);
        sizesum *= dimPool[randomdim];
        // llvm::outs()<<dimPool[randomdim]<<" ";
      } else {
        dims.push_back(1);
        // llvm::outs()<<1<<" ";
      }
    }
    // llvm::outs()<<"\n";
    shapePool.push_back(dims);
  }
}

inline mlir::Type randomIntOrFloatType(mlir::MLIRContext *ctx) {
  auto supportedTypes = getSupportedIntOrFloatTypes(ctx);
  return supportedTypes[random(supportedTypes.size())];
}

inline mlir::Type randomElementaryOrIndexType(mlir::MLIRContext *ctx) {
  auto types = getSupportedIntOrFloatTypes(ctx);
  types.push_back(mlir::IndexType::get(ctx));
  return types[random(types.size())];
}

inline mlir::VectorType randomVectorType(mlir::MLIRContext *ctx) {
  auto elemTy = randomIntOrFloatType(ctx);
  auto shape = shapePool[random(shapePool.size())];
  return mlir::VectorType::get(shape, elemTy);
}

inline mlir::VectorType randomVectorType(mlir::Type elemTy) {
  auto shape = shapePool[random(shapePool.size())];
  return mlir::VectorType::get(shape, elemTy);
}

inline mlir::VectorType random1DVectorType(mlir::Type elemTy) {
  llvm::SmallVector<int64_t> shape;
  shape.push_back(dimPool[random(dimPool.size())]);
  return mlir::VectorType::get(shape, elemTy);
}

inline mlir::VectorType random1DVectorType(mlir::MLIRContext *ctx) {
  llvm::SmallVector<int64_t> shape;
  shape.push_back(dimPool[random(dimPool.size())]);
  return mlir::VectorType::get(shape, randomIntOrFloatType(ctx));
}

// memref can be static ranked, dynamically ranked and unranked.
inline mlir::MemRefType randomStaticShapedMemrefType(mlir::MLIRContext *ctx) {
  auto elemType = randomIntOrFloatType(ctx);
  auto shape = shapePool[random(shapePool.size())];
  return mlir::MemRefType::get(shape, elemType);
}

inline mlir::MemRefType randomStaticShapedMemrefType(mlir::Type elemTy) {
  auto shape = shapePool[random(shapePool.size())];
  return mlir::MemRefType::get(shape, elemTy);
}

inline mlir::MemRefType randomDynamicShapedMemrefType(mlir::MLIRContext *ctx) {
  auto elemType = randomIntOrFloatType(ctx);
  auto shape = shapePool[random(shapePool.size())];
  auto num = random(shape.size()) + 1;
  for (int i = 0; i < num; ++i) {
    shape[i] = mlir::ShapedType::kDynamic;
  }
  return mlir::MemRefType::get(shape, elemType);
}

inline mlir::MemRefType randomDynamicShapedMemrefType(mlir::Type elemType) {
  auto shape = shapePool[random(shapePool.size())];
  auto num = random(shape.size()) + 1;
  for (int i = 0; i < num; ++i) {
    shape[i] = mlir::ShapedType::kDynamic;
  }
  return mlir::MemRefType::get(shape, elemType);
}

inline mlir::MemRefType randomRankedMemrefType(mlir::MLIRContext *ctx) {
  if (UR(2)) {
    return randomStaticShapedMemrefType(ctx);
  } else {
    return randomDynamicShapedMemrefType(ctx);
  }
}

inline mlir::MemRefType randomRankedMemrefType(mlir::Type ty) {
  if (UR(2)) {
    return randomStaticShapedMemrefType(ty);
  } else {
    return randomDynamicShapedMemrefType(ty);
  }
}

inline mlir::RankedTensorType randomStaticShapedTensorType(mlir::MLIRContext *ctx) {
  auto elemType = randomIntOrFloatType(ctx);
  auto shape = shapePool[random(shapePool.size())];
  return mlir::RankedTensorType::get(shape, elemType);
}

inline mlir::RankedTensorType randomStaticShapedTensorType(mlir::Type elemTy) {
  auto shape = shapePool[random(shapePool.size())];
  return mlir::RankedTensorType::get(shape, elemTy);
}

inline mlir::RankedTensorType randomDynamicShapedTensorType(mlir::MLIRContext *ctx) {
  auto elemType = randomIntOrFloatType(ctx);
  auto shape = shapePool[random(shapePool.size())];
  auto num = random(shape.size()) + 1;
  for (int i = 0; i < num; ++i) {
    shape[i] = mlir::ShapedType::kDynamic;
  }
  return mlir::RankedTensorType::get(shape, elemType);
}

inline mlir::RankedTensorType randomDynamicShapedTensorType(mlir::Type elemTy) {
  auto shape = shapePool[random(shapePool.size())];
  auto num = random(shape.size()) + 1;
  for (int i = 0; i < num; ++i) {
    shape[i] = mlir::ShapedType::kDynamic;
  }
  return mlir::RankedTensorType::get(shape, elemTy);
}

inline mlir::RankedTensorType randomRankedTensorType(mlir::MLIRContext *ctx) {
  if (UR(2)) {
    return randomStaticShapedTensorType(ctx);
  } else {
    return randomDynamicShapedTensorType(ctx);
  }
}

inline mlir::ShapedType randomStaticShapeType(mlir::Type elemTy) {
  switch (random(3)) {
  case 0:
    return randomStaticShapedMemrefType(elemTy);
  case 1:
    return randomStaticShapedTensorType(elemTy);
  default:
    return randomVectorType(elemTy);
  }
}

inline mlir::ShapedType randomStaticShapeType(mlir::MLIRContext *ctx) {
  auto elemTy = randomIntOrFloatType(ctx);
  return randomStaticShapeType(elemTy);
}

inline mlir::Type randomIntegerType(mlir::MLIRContext *ctx) {
  std::vector<mlir::Type> candidates;
  auto supportedTypes = getSupportedIntTypes(ctx);
  auto elemTy = supportedTypes[random(supportedTypes.size())];
  candidates.push_back(elemTy);
  candidates.push_back(randomStaticShapedMemrefType(elemTy));
  candidates.push_back(randomStaticShapedTensorType(elemTy));
  return candidates[random(candidates.size())];
}

inline mlir::Type randomFloatType(mlir::MLIRContext *ctx) {
  std::vector<mlir::Type> candidates;
  auto supportedTypes = getSupportedFloatTypes(ctx);
  auto elemTy = supportedTypes[random(supportedTypes.size())];
  candidates.push_back(elemTy);
  candidates.push_back(randomStaticShapedMemrefType(elemTy));
  candidates.push_back(randomStaticShapedTensorType(elemTy));
  return candidates[random(candidates.size())];
}

inline mlir::Type randomType(mlir::MLIRContext *ctx) {
  std::vector<mlir::Type> candidates = {mlir::IndexType::get(ctx),
                                  randomIntOrFloatType(ctx),
                                  randomStaticShapedMemrefType(ctx),
                                  randomStaticShapedTensorType(ctx),
                                  randomDynamicShapedTensorType(ctx),
                                  randomDynamicShapedMemrefType(ctx),
                                  randomVectorType(ctx)};
  return candidates[random(candidates.size())];
}

inline mlir::Type randomNonTensorType(mlir::MLIRContext *ctx) {
  std::vector<mlir::Type> candidates = {randomIntOrFloatType(ctx),
                                  randomStaticShapedMemrefType(ctx)};
  return candidates[random(candidates.size())];
}

inline mlir::FunctionType randomFunctionType(mlir::MLIRContext *ctx) {
  //  int func_arg_num = random(func_arg_ub);
  //  int func_ret_num = random(func_ret_ub);

  int func_arg_num = 0; // random(func_arg_ub);
  int func_ret_num = 0; // random(func_ret_ub);

  llvm::SmallVector<mlir::Type> argTypes;
  llvm::SmallVector<mlir::Type> retTypes;
  for (int i = 0; i < func_arg_num; ++i) {
    argTypes.push_back(randomType(ctx));
  }
  for (int i = 0; i < func_ret_num; ++i) {
    retTypes.push_back(randomType(ctx));
  }
  auto funcType =
      mlir::FunctionType::get(ctx, mlir::TypeRange(argTypes), mlir::TypeRange(retTypes));
  return funcType;
}

inline mlir::Type randomMemrefOrRankedTensorType(mlir::MLIRContext *ctx) {
  auto elemTy = randomIntOrFloatType(ctx);
  auto shape = shapePool[random(shapePool.size())];
  if (random(2)) {
    return mlir::RankedTensorType::get(shape, elemTy);
  } else {
    return mlir::MemRefType::get(shape, elemTy);
  }
}

inline mlir::Type randomMemrefOrRankedTensorType(mlir::ShapedType t) {
  auto shape = t.getShape();
  auto elemTy = t.getElementType();
  if (random(2)) {
    return mlir::RankedTensorType::get(shape, elemTy);
  } else {
    return mlir::MemRefType::get(shape, elemTy);
  }
}

#endif // MLIR_FUZZ_TYPEGENERATION_H
