#include <cmath>
#include <random>
#include <string>

#include "mlir/Dialect/Affine/Passes.h"
// #include "mlir/Dialect/Index/IR/IndexEnums.h.inc"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/InitAllDialects.h"

#include "fixub/ConstantPool.h"
#include "fixub/DelOp.h"
#include "fixub/FixTensor.h"
#include "fixub/FixUBUtils.h"

void FixTensor::fixtensor(mlir::Operation *op) {
  std::string opname = op->getName().getStringRef().str();
  if (opname == "tensor.empty") {
    FixTensor::fixEmpty(op);
  }
  if (opname == "tensor.cast") {
    FixTensor::fixCast(op);
  }
  if (opname == "tensor.dim") {
    FixTensor::fixDim(op);
  }
  if (opname == "tensor.scatter") {
    FixTensor::fixScatter(op);
  }
  if (opname == "tensor.unpack") {
    FixTensor::fixUnpack(op);
  }
  if (opname == "tensor.pack") {
    FixTensor::fixPack(op);
  }
  if (opname == "tensor.extract") {
    FixTensor::fixExtract(op);
  }
  if (opname == "tensor.insert") {
    FixTensor::fixInsert(op);
  }
  return;
}

// void FixTensor::fixCast(mlir::Operation *op) {
//   /**
//    * Fix tensor.cast operation, which may implicitly expand the element
//    number
//    * of operand tensor.
//    */
//   llvm::outs() << "[FixTensor::fixCast] start\n";

//   mlir::Location loc = op->getLoc();
//   mlir::OpBuilder builder(op->getBlock(), op->getBlock()->begin());
//   builder.setInsertionPoint(op);

//   mlir::Value oriTensor = op->getOperand(0);
//   mlir::Value castedTensor = op->getResult(0);

// 	auto elementTy =
// llvm::cast<mlir::ShapedType>(oriTensor.getType()).getElementType();
//   if(llvm::isa<mlir::FloatType>(elementTy))
//       return ;

//   mlir::ShapedType oriTy = oriTensor.getType().dyn_cast<mlir::ShapedType>();
//   mlir::ShapedType castedTy =
//       castedTensor.getType().dyn_cast<mlir::ShapedType>();

//   assert(oriTy.hasRank());
//   assert(castedTy.hasRank());

//   llvm::ArrayRef<int64_t> oriShape = oriTy.getShape();
//   llvm::ArrayRef<int64_t> castedShape = castedTy.getShape();

//   llvm::outs() << "[FixTensor::fixCast] calculate the element number of "
//                   "original tensor and casted tensor.\n";
//   llvm::SmallVector<int64_t> oriStaticShape;
//   llvm::SmallVector<mlir::Value> oriDynamicShape;
//   llvm::SmallVector<int64_t> castedStaticShape;
//   llvm::SmallVector<mlir::Value> castedDynamicShape;
//   // get static and dynamic dimension size of origianl tensor
//   for (int64_t i = 0; i < oriTy.getRank(); i++) {
//     if (oriTy.isDynamic(i) || oriTy.getDimSize(i) <= 0) {
//       llvm::outs() << "[FixTensor::fixCast] collect original tensor type: "
//                       "dynamic branch\n";
//       mlir::Value currentIdx =
//           ConstantPool::getValue(i, builder.getIndexType());
//       mlir::Value curDimSize =
//           builder.create<mlir::tensor::DimOp>(loc, oriTensor, currentIdx)
//               .getOperation()
//               ->getResult(0);
//       oriDynamicShape.push_back(curDimSize);
//     } else {
//       llvm::outs() << "[FixTensor::fixCast] collect original tensor type: "
//                       "static branch\n";
//       oriStaticShape.push_back(oriTy.getDimSize(i));
//     }
//   }
//   // get static and dynamic dimension size of casted tensor.
//   for (int64_t i = 0; i < castedTy.getRank(); i++) {
//     if (castedTy.isDynamic(i) || castedTy.getDimSize(i) <= 0) {
//       llvm::outs() << "[FixTensor::fixCast] collect casted tensor type: "
//                       "dynamic branch\n";
//       mlir::Value currentIdx =
//           ConstantPool::getValue(i, builder.getIndexType());
//       mlir::Value curDimSize =
//           builder.create<mlir::tensor::DimOp>(loc, castedTensor, currentIdx)
//               .getOperation()
//               ->getResult(0);
//       oriDynamicShape.push_back(curDimSize);
//     } else {
//       llvm::outs()
//           << "[FixTensor::fixCast] collect casted tensor type: static
//           branch\n";
//       oriStaticShape.push_back(castedTy.getDimSize(i));
//     }
//   }
//   // calculate the shape of original and casted tensors.
//   int64_t oriStaticElemNum = 1;
//   for (auto i : oriStaticShape) {
//     oriStaticElemNum *= i;
//   }
//   mlir::Value oriStaticElemNumVal =
//       ConstantPool::getValue(oriStaticElemNum, builder.getIndexType());
//   oriDynamicShape.push_back(oriStaticElemNumVal);
//   mlir::Value oriElemNumVal = ConstantPool::getValue(1,
//   builder.getIndexType()); for (auto val : oriDynamicShape) {
//     oriElemNumVal =
//         builder.create<mlir::arith::MulIOp>(loc, oriElemNumVal, val);
//   }
//   // calculate the shape of original and casted tensors end, oriDynamicShape
//   is
//   // the value (in runtime).

//   int64_t castedStaticElemNum = 1;
//   for (auto i : castedStaticShape) {
//     castedStaticElemNum *= i;
//   }
//   mlir::Value castedStaticElemNumVal =
//       ConstantPool::getValue(castedStaticElemNum, builder.getIndexType());
//   castedDynamicShape.push_back(castedStaticElemNumVal);
//   mlir::Value castedElemNumVal =
//       ConstantPool::getValue(1, builder.getIndexType());
//   for (auto val : castedDynamicShape) {
//     castedElemNumVal =
//         builder.create<mlir::arith::MulIOp>(loc, castedElemNumVal, val);
//   }

//   llvm::outs()
//       << "[FixTensor::fixCast] dynamically compare the element number.\n";
//   // mlir::Value invalidShape =
//   // builder.create<mlir::arith::CmpIOp>(op->getLoc(),
//   // mlir::arith::CmpIPredicate::uge, oriElemNumVal,
//   // castedElemNumVal).getOperation()->getResult(0);
//   mlir::Value invalidShape =
//       builder
//           .create<mlir::arith::CmpIOp>(op->getLoc(),
//                                        mlir::arith::CmpIPredicate::ne,
//                                        oriElemNumVal, castedElemNumVal)
//           .getOperation()
//           ->getResult(0);
//   auto invalidBranchBuilder = [&](mlir::OpBuilder &b, mlir::Location loc) {
//     llvm::outs() << "[FixTensor::fixCast] num(ori) >= num(casted).\n";
//     // generate a valid operand that equils to the required one.
//     // the result type and use splat to get it.
//     int64_t bitwidth = castedTy.getElementType().getIntOrFloatBitWidth();
//     int64_t randnum = FixUBUtils::randsval(bitwidth);
//     mlir::Value randnumVal =
//         builder
//             .create<mlir::arith::ConstantOp>(
//                 loc, mlir::IntegerAttr::get(castedTy.getElementType(),
//                 randnum))
//             .getOperation()
//             ->getResult(0);
//     mlir::Value res =
//         builder.create<mlir::tensor::SplatOp>(loc, castedTy, randnumVal)
//             .getOperation()
//             ->getResult(0);
//     b.create<mlir::scf::YieldOp>(loc, res);
//   };
//   auto validBranchBuilder = [&](mlir::OpBuilder &b, mlir::Location loc) {
//     llvm::outs() << "[FixTensor::fixCast] num(ori) < num(casted).\n";
//     b.create<mlir::scf::YieldOp>(loc, oriTensor);
//   };
//   mlir::Value newOriTensor =
//       builder
//           .create<mlir::scf::IfOp>(loc, invalidShape, invalidBranchBuilder,
//                                    validBranchBuilder)
//           .getOperation()
//           ->getResult(0);

//   llvm::outs() << "[FixTensor::fixCast] replace the operand.\n";
//   op->setOperand(0, newOriTensor);

//   llvm::outs() << "[FixTensor::fixCast] end\n";
//   return;
// }

void FixTensor::fixEmpty(mlir::Operation *op) {
  llvm::outs() << "[FixTensor::fixEmpty] start\n";
  mlir::Location loc = op->getLoc();
  mlir::OpBuilder builder(op->getBlock(), op->getBlock()->begin());
  builder.setInsertionPoint(op);

  mlir::Value tensor = op->getResult(0);

  auto elementTy =
      llvm::cast<mlir::ShapedType>(tensor.getType()).getElementType();
  // if (llvm::isa<mlir::FloatType>(elementTy))
  //   return;

  mlir::ShapedType tensorTy = tensor.getType().dyn_cast<mlir::ShapedType>();

  mlir::Value newVal =
      FixUBUtils::genRandomTensorFromExisting(tensorTy, builder, loc);
  // replace this operation
  op->replaceAllUsesWith(newVal.getDefiningOp());
  delOps.push_back(op);
  llvm::outs() << "[FixTensor::fixEmpty] end\n";
  return;
}

void FixTensor::fixDim(mlir::Operation *op) {
  /**
   * This function fixes undefined behavior in tensor.dim.
   * The documentation about UB says:
   * If the dimension index is out of bounds, the behavior is undefined.
   *
   * Consider the following example:
   * %dim_size = tensor.dim %tensor, %dim : tensor<4x?xf32>
   * If %dim is higher than 2, then the undefined behavoir occurs.
   *
   * For fix this, %dim should compare to the rank of %tensor,
   * and do %dim = %dim % rank(%tensor)
   */
  mlir::Location loc = op->getLoc();
  mlir::OpBuilder builder(op->getBlock(), op->getBlock()->begin());
  builder.setInsertionPoint(op);

  mlir::Value tensor = op->getOperand(0), dim = op->getOperand(1);

  auto elementTy =
      llvm::cast<mlir::ShapedType>(tensor.getType()).getElementType();

  // FlaotType should not be banded, because tensor.dim got index type value
  // if(llvm::isa<mlir::FloatType>(elementTy))
  //     return ;

  auto rank = builder.create<mlir::tensor::RankOp>(loc, tensor);
  auto comparisonOp = builder.create<mlir::index::CmpOp>(
      op->getLoc(), mlir::index::IndexCmpPredicate::UGE, dim, rank);
  auto GreaterBuilder = [&](mlir::OpBuilder &b,
                            mlir::Location loc) { // do dim % rank
    mlir::Operation *newDimOp = b.create<mlir::index::RemUOp>(loc, dim, rank);
    mlir::Value newDim = newDimOp->getResult(0);
    b.create<mlir::scf::YieldOp>(loc, newDim);
  };
  auto NotGreaterBuilder =
      [&](mlir::OpBuilder &b,
          mlir::Location loc) { // return original dimension directly
        b.create<mlir::scf::YieldOp>(loc, dim);
      };
  auto ifOp =
      builder.create<mlir::scf::IfOp>(op->getLoc(), comparisonOp->getResult(0),
                                      GreaterBuilder, NotGreaterBuilder);
  auto newDim = ifOp->getResult(0);
  op->setOperand(1, newDim);
  return;
}

mlir::Operation *FixTensor::findNotUseVal(
    int64_t curIndicesDim, // currrent dimension that scf.for reaches.
    int64_t rank,          // the rank of traversed tensor(the %indices)
    llvm::SmallVector<mlir::Value>
        indicesIdxs, // the current index of traversed tensor(the %indices)
    mlir::ValueRange
        iterArgs, // the value that have been effected by the scf.for
    mlir::Location loc, mlir::OpBuilder builder) {
  // TODO:
  mlir::Operation *scfForOp;

  llvm::outs() << scfForOp << "\n";

  if (curIndicesDim < rank) { // traverse until reaches the last dimension
    llvm::outs() << "[FixTensor::findNotUseVal] scf outter loop start\n";

    mlir::Value tensorVal = iterArgs[0];
    mlir::Value lb = ConstantPool::getValue(0, builder.getIndexType());
    mlir::Value curIndicesDimVal =
        ConstantPool::getValue(curIndicesDim, builder.getIndexType());
    auto ub =
        builder.create<mlir::tensor::DimOp>(loc, tensorVal, curIndicesDimVal);
    mlir::Value step = ConstantPool::getValue(1, builder.getIndexType());
    auto blockBuilder = [&](mlir::OpBuilder &builder, mlir::Location loc,
                            mlir::Value iv /*loop iterator*/,
                            mlir::ValueRange args) {
      indicesIdxs.push_back(iv);
      mlir::Operation *innerForOp = findNotUseVal(
          curIndicesDim + 1, rank, indicesIdxs, args, loc, builder);
      if (curIndicesDim + 1 < rank) {
        assert(innerForOp);
        builder.create<mlir::scf::YieldOp>(loc, innerForOp->getResults());
      }
    };
    scfForOp = builder.create<mlir::scf::ForOp>(loc, lb, ub, step, iterArgs,
                                                blockBuilder);

    llvm::outs() << "[FixTensor::findNotUseVal] scf outter loop end\n";
  } else {
    llvm::outs() << "[FixTensor::findNotUseVal] scf inner loop start\n";

    mlir::Value table = iterArgs[0];
    llvm::SmallVector<mlir::Value> oldRes;
    for (int64_t i = 1; i < iterArgs.size(); i++) {
      oldRes.push_back(iterArgs[i]);
    }

    auto used =
        builder.create<mlir::tensor::ExtractOp>(loc, table, indicesIdxs);
    auto usedBuilder = [&](mlir::OpBuilder &b, mlir::Location loc) {
      llvm::outs() << "[FixTensor::findNotUseVal] indices used branch, return "
                      "old indices directly.\n";
      b.create<mlir::scf::YieldOp>(loc, oldRes);
    };
    auto notUsedBuilder = [&](mlir::OpBuilder &b, mlir::Location loc) {
      llvm::outs() << "[FixTensor::findNotUseVal] indices not used branch, "
                      "return current indices.\n";
      b.create<mlir::scf::YieldOp>(loc, indicesIdxs);
    };
    mlir::Operation *ifOp =
        builder.create<mlir::scf::IfOp>(loc, used, usedBuilder, notUsedBuilder);

    auto newRes = ifOp->getResults();

    llvm::SmallVector<mlir::Value> res;
    res.push_back(table);
    for (auto nr : newRes) {
      res.push_back(nr);
    }
    builder.create<mlir::scf::YieldOp>(loc, res);

    llvm::outs() << "[FixTensor::findNotUseVal] scf inner loop end\n";
  }
  llvm::outs() << "[FixTensor::findNotUseVal] end\n";
  llvm::outs() << scfForOp << "\n";
  return scfForOp;
}

mlir::Operation *FixTensor::traverseScatterIndices(
    mlir::tensor::ScatterOp scatterOp, // the related scatter operation
    int64_t curIndicesDim, // currrent dimension that scf.for reaches.
    int64_t rank,          // the rank of traversed tensor(the %indices)
    llvm::SmallVector<mlir::Value>
        indicesIdxs, // the current index of traversed tensor(the %indices)
    mlir::ValueRange
        iterArgs, // the value that have been effected by the scf.for
    mlir::Location loc, mlir::OpBuilder builder) {
  // tarnverse
  mlir::Operation *forOp;

  llvm::outs() << "[FixTensor::traverseScatterIndices]\n";
  if (curIndicesDim < rank - 1) { // traverse until reaches the last dimension
    llvm::outs() << "[FixTensor::traverseScatterIndices] scf outter loop\n";
    mlir::Value tensorVal = iterArgs[0];
    mlir::Value lb = ConstantPool::getValue(0, builder.getIndexType());
    mlir::Value curIndicesDimVal =
        ConstantPool::getValue(curIndicesDim, builder.getIndexType());
    auto ub =
        builder.create<mlir::tensor::DimOp>(loc, tensorVal, curIndicesDimVal);
    mlir::Value step = ConstantPool::getValue(1, builder.getIndexType());
    auto blockBuilder = [&](mlir::OpBuilder &builder, mlir::Location loc,
                            mlir::Value iv /*loop iterator*/,
                            mlir::ValueRange args) {
      indicesIdxs.push_back(iv);
      mlir::Operation *innerForOp = traverseScatterIndices(
          scatterOp, curIndicesDim + 1, rank, indicesIdxs, args, loc, builder);
      if (curIndicesDim + 1 < rank - 1) {
        assert(innerForOp);
        builder.create<mlir::scf::YieldOp>(loc, innerForOp->getResults());
      }
    };
    forOp = builder.create<mlir::scf::ForOp>(loc, lb, ub, step, iterArgs,
                                             blockBuilder);
  } else {
    mlir::Value indices = iterArgs[0];
    mlir::Value table = iterArgs[1];

    llvm::outs() << "[FixTensor::traverseScatterIndices] scf last dimension\n";

    llvm::outs() << "[FixTensor::traverseScatterIndices] get all values of the "
                    "last dimension of \%indices\n";
    llvm::SmallVector<mlir::Value> lastDimVals;
    llvm::ArrayRef<int64_t> scatterDims = scatterOp.getScatterDims();
    int64_t indicesLastDim =
        scatterDims.size(); // len(scatter_dims) == last dimension of %indices
    for (int64_t i = 0; i < indicesLastDim; i++) {
      auto lastDimIdx = ConstantPool::getValue(i, builder.getIndexType());
      indicesIdxs.push_back(lastDimIdx);
      auto indicesVal =
          builder.create<mlir::tensor::ExtractOp>(loc, indices, indicesIdxs);
      indicesIdxs.pop_back();
      lastDimVals.push_back(indicesVal);
    }

    llvm::outs() << "[FixTensor::traverseScatterIndices] check whether the "
                    "index has been used, and find the unused index.\n";
    auto cond =
        builder.create<mlir::tensor::ExtractOp>(loc, table, lastDimVals);
    auto usedBuilder = [&](mlir::OpBuilder &builder, mlir::Location loc) {
      llvm::outs() << "[FixTensor::traverseScatterIndices] if it has already "
                      "been used, find a false value in table, and return.\n";
      llvm::SmallVector<mlir::Value> curIdx;
      llvm::SmallVector<mlir::Value> iterArgs;
      iterArgs.push_back(table);
      mlir::ShapedType tableTy = table.getType().dyn_cast<mlir::ShapedType>();
      for (int64_t i = 0; i < tableTy.getRank();
           i++) { // If it has been used, we use 0 to initilize.
        auto idx = ConstantPool::getValue(0, builder.getIndexType());
        iterArgs.push_back(idx);
      }
      llvm::outs() << "[FixTensor::traverseScatterIndices] find a new index "
                      "that has not been used.\n";
      mlir::Operation *scfForOp =
          findNotUseVal(0, tableTy.getRank(), curIdx, iterArgs, loc, builder);
      assert(scfForOp); // the result is table and indices
      builder.create<mlir::scf::YieldOp>(loc, scfForOp->getResults());
    };
    auto notUsedBuilder = [&](mlir::OpBuilder &builder, mlir::Location loc) {
      llvm::outs() << "[FixTensor::traverseScatterIndices] if it has not been "
                      "used, directly return the unused value(s).\n";
      llvm::SmallVector<mlir::Value> iterArgs;
      int64_t indicesLastDim =
          scatterDims.size(); // len(scatter_dims) == last dimension of %indices
      iterArgs.push_back(table);
      for (int64_t i = 0; i < indicesLastDim;
           i++) { // If it has not been used, return original indices directly.
        auto lastDimIdx = ConstantPool::getValue(i, builder.getIndexType());
        indicesIdxs.push_back(lastDimIdx);
        auto indicesVal =
            builder.create<mlir::tensor::ExtractOp>(loc, indices, indicesIdxs);
        indicesIdxs.pop_back();
        iterArgs.push_back(indicesVal);
      }
      builder.create<mlir::scf::YieldOp>(loc, iterArgs);
    };
    llvm::outs() << "[FixTensor::traverseScatterIndices] use scf.if to return "
                    "this value.\n";
    auto ifOp =
        builder.create<mlir::scf::IfOp>(loc, cond, usedBuilder, notUsedBuilder);

    llvm::outs()
        << "[FixTensor::traverseScatterIndices] change the table value.\n";
    auto ifRes = ifOp->getResults();
    table = ifRes[0];
    llvm::SmallVector<mlir::Value> newIndices;
    for (int64_t i = 1; i < ifRes.size(); i++) {
      newIndices.push_back(ifRes[i]);
    }
    auto trueVal = ConstantPool::getValue(1, builder.getI1Type());
    mlir::Value newTable =
        builder.create<mlir::tensor::InsertOp>(loc, trueVal, table, newIndices)
            ->getResult(0);

    llvm::outs()
        << "[FixTensor::traverseScatterIndices] fix the indices value.\n";
    for (int64_t i = 0; i < indicesLastDim; i++) {
      auto lastDimIdx = ConstantPool::getValue(i, builder.getIndexType());
      indicesIdxs.push_back(lastDimIdx);
      indices = builder
                    .create<mlir::tensor::InsertOp>(loc, newIndices[i], indices,
                                                    indicesIdxs)
                    .getOperation()
                    ->getResult(0);
      indicesIdxs.pop_back();
    }

    llvm::outs() << "[FixTensor::traverseScatterIndices] yeild all effected "
                    "values in scf.for.\n";
    llvm::SmallVector<mlir::Value> forRes;
    forRes.push_back(indices);
    forRes.push_back(newTable);
    builder.create<mlir::scf::YieldOp>(loc, forRes);
  }
  return forOp;
}

mlir::Operation *
FixTensor::fixScatterUniqueIndices(mlir::tensor::ScatterOp scatterOp,
                                   mlir::Location loc,
                                   mlir::OpBuilder builder) {
  // traverse the indices, until reaches the last dimension

  // mlir::tensor::ScatterOp scatterOp,  // the related scatter operation
  // int64_t curIndicesDim,  // currrent dimension that scf.for reaches.
  // int64_t rank,   // the rank of traversed tensor(the %indices)
  // llvm::SmallVector<mlir::Value> indicesIdxs,   // the current index of
  // traversed tensor(the %indices) mlir::ValueRange iterArgs,  // the value
  // that have been effected by the scf.for mlir::Location loc, mlir::OpBuilder
  // builder
  mlir::Operation *op = scatterOp.getOperation();

  mlir::Value dest = op->getOperand(1);
  mlir::Value indices = op->getOperand(2);
  llvm::ArrayRef<int64_t> scatterDims = scatterOp.getScatterDims();

  mlir::ShapedType indexTy = indices.getType().dyn_cast<mlir::ShapedType>();
  mlir::ShapedType destTy = dest.getType().dyn_cast<mlir::ShapedType>();

  llvm::ArrayRef<int64_t> destShape = destTy.getShape();
  llvm::ArrayRef<int64_t> indexShape = indexTy.getShape();

  llvm::SmallVector<mlir::Value> indicesIdxs;

  llvm::outs() << "[FixTensor::fixScatterUniqueIndices] generate a new table "
                  "according to the existing indices.\n";
  // 1. indices 2. table
  // TODO: create the table
  int64_t tableSize = 1;
  llvm::SmallVector<mlir::Value> iterArgs;
  llvm::SmallVector<int64_t> tableShape;
  llvm::SmallVector<mlir::Value> tableElem;
  for (int64_t i = 0; i < scatterDims.size(); i++) {
    tableShape.push_back(destShape[scatterDims[i]]);
    tableSize *= destShape[scatterDims[i]];
  }
  mlir::RankedTensorType tableTy =
      mlir::RankedTensorType::get(tableShape, builder.getI1Type());
  for (int64_t i = 0; i < tableSize; i++) {
    tableElem.push_back(ConstantPool::getValue(0, builder.getI1Type()));
  }
  mlir::Value table = builder.create<mlir::tensor::FromElementsOp>(
      loc, tableTy, mlir::ValueRange(tableElem));
  iterArgs.push_back(indices);
  iterArgs.push_back(table);

  mlir::Operation *forOp = traverseScatterIndices(
      scatterOp, 0, indexShape.size(), indicesIdxs, iterArgs, loc, builder);
  assert(forOp);
  return forOp;
}

void FixTensor::fixScatter(mlir::Operation *op) {
  /**
   * Note: the source of tensor dialect is located in
   * build_bak/tools/mlir/include/mlir/Dialect/Tensor/IR/TensorOps.cpp.inc,
   * which is generated from .td files.
   *
   * This sfunction fixes undefined behavior in tensor.scatter.
   * The documentation says:
   * The indices are expected to be confined to coordinate values that fit the
   * range of the dest tensor, otherwise the behavior is undefined. A unique
   * unit attribute must be be specified to indicate that the coordinates are
   * statically guaranteed to be unique at runtime. If coordinates are not truly
   * unique at runtime, the behavior is undefined.
   *
   * To fix ubs in this operation, we should:
   * 1. Ensure that tensor.dim(%dest[%scatter_dims[i]]) >
   * %index[*][%scatter_dims[i]], where * is for any leading dimensions.
   * 2. Ensure that all of the %index[*][%scatter_dims[i]] do not have same
   * element value.
   *
   * The corresponding test: test/tensor.py/static_shape_tensor_test()
   */
  llvm::outs() << "[FixTensor::fixScatter] start\n";

  // Get the scatter dims, %src, %dest, and %indices.
  mlir::Location loc = op->getLoc();
  mlir::OpBuilder builder(op->getBlock(), op->getBlock()->begin());
  builder.setInsertionPoint(op);

  mlir::tensor::ScatterOp scatterOp =
      llvm::dyn_cast<mlir::tensor::ScatterOp>(op);
  bool unique = scatterOp.getUnique();
  llvm::ArrayRef<int64_t> scatterDims = scatterOp.getScatterDims();
  mlir::Value src = op->getOperand(0);
  mlir::Value dest = op->getOperand(1);
  mlir::Value indices = op->getOperand(2);

  auto elementTy = llvm::cast<mlir::ShapedType>(src.getType()).getElementType();
  if (llvm::isa<mlir::FloatType>(elementTy))
    return;

  mlir::ShapedType srcTy = src.getType().dyn_cast<mlir::ShapedType>();
  mlir::ShapedType destTy = dest.getType().dyn_cast<mlir::ShapedType>();
  mlir::ShapedType indexTy = indices.getType().dyn_cast<mlir::ShapedType>();

  // TODO: check all dimensions of all tensors.
  // if the corresponding dimension is ?, getShape will return
  // -9223372036854775808. I think dynamic dimension should not appear in the
  // calculation... Well, will shape propagation solve this problem?
  assert(srcTy.hasStaticShape());
  assert(destTy.hasStaticShape());
  assert(indexTy.hasStaticShape());
  // ... I think this for a long time, and I think that we can use tensor.dim to
  // handle this.

  llvm::ArrayRef<int64_t> srcShape = srcTy.getShape();
  llvm::ArrayRef<int64_t> destShape = destTy.getShape();
  llvm::ArrayRef<int64_t> indexShape = indexTy.getShape();

  mlir::Type srcElemTy = srcTy.getElementType();
  mlir::Type destElemTy = destTy.getElementType();
  mlir::Type indicesElemTy = indexTy.getElementType();

  // Condition1 [Checked by parser]: length of %scatter_dim must be smaller than
  // the rank of %dest, that is already checked by parser. Condition2 [Checked
  // by parser]: each element of %scatter_dim must be in range of the rank of
  // %dest, that is already checked by parser. Condition3 [Checked by parser]:
  // rank(%source) = (rank(%index)-1) + (rank(%dest)-len(scatter_dim)), that is
  // already checked by parser. Condition4 [Checked by parser]: the
  // corresponding demensions in %source must be 1, if it appears in
  // scatter_dim, that is already checked by parser.

  // now I think, all dynamic dimensions can be cast to static, and do following
  // check!

  // Condition5 [TODO]: the corresponding demensions in %source must be as same
  // as %dest, if it does not appear in scatter_dim.
  //   This condition will happen if corresponding dimensions of %source and
  //   %dest have dynamic dimension.
  // Condition6 [TODO]: the leading dimensions of %source should be as same as
  // the leading dimensions of %index.
  //   This condition will happen if corresponding dimensions of %source and
  //   %index have dynamic dimension.

  // Condition8: the elements in the last dimensions of %index should be in the
  // range of corresponding dimensions of %dest in scatter_dim
  //   This condition can be achieved easily.
  std::vector<int64_t> destDimSizeInScatter = std::vector<int64_t>();
  for (auto scatterD : scatterDims) {
    destDimSizeInScatter.push_back(destShape[scatterD]);
  }
  int64_t leadingDims = 1;
  for (auto indexD : indexShape) {
    leadingDims *= indexD;
  }
  assert(leadingDims % scatterDims.size() == 0);
  leadingDims /= scatterDims.size(); // because that len(scatter_dims) == last
                                     // dimension size of index.
  llvm::SmallVector<mlir::Value> indicesMaskElements;
  for (int64_t i = 0; i < leadingDims; i++) {
    for (auto dimSize : destDimSizeInScatter) {
      assert(dimSize >= 0);
      auto d = builder.create<mlir::arith::ConstantOp>(
          loc, mlir::IntegerAttr::get(indicesElemTy, dimSize));
      indicesMaskElements.push_back(d); // we assume that there is no 0
    }
  }
  mlir::RankedTensorType indicesMaskTy =
      mlir::RankedTensorType::get(indexShape, indicesElemTy);
  auto indicesMask = builder.create<mlir::tensor::FromElementsOp>(
      loc, indicesMaskTy, mlir::ValueRange(indicesMaskElements));
  auto fixedIndices =
      builder.create<mlir::arith::RemUIOp>(loc, indices, indicesMask);
  op->setOperand(2, fixedIndices);

  // Condition7: the elements in the last dimensions of %index should not be
  // duplicate if unique attribute is set.
  //   This condition can be achieved easily.
  if (unique) {
    // do not as same as xxx.
    mlir::Operation *forOp = fixScatterUniqueIndices(scatterOp, loc, builder);
    mlir::Value fixedIndices = forOp->getResult(0);
    op->setOperand(2, fixedIndices);
  }

  // Condition9 [TODO]: the last dimensions of %index should be as same as the
  // length of scatter_dim.
  //   This condition will happend if the last demension of %index is dynamic.

  llvm::outs() << "[FixTensor::fixScatter] end\n";
  return;
}

// The undefined behaviors in these two operations are checked in semantic
// checks.
void FixTensor::fixUnpack(mlir::Operation *op) { return; }

void FixTensor::fixPack(mlir::Operation *op) { return; }

void FixTensor::fixCast(mlir::Operation *op) {
  /**
   * Fix undefined behaviors in tensor.cast.
   *
   * Example:
   * %cast = tensor.cast %5 : tensor<?x?x?xi64> to tensor<15x?x16xi64>
   *
   * This fix work does the following tasks:
   * (1) Fix the mle problem caused by the dimension dismatch between
   * src dynamic dim and dest static dim
   *
   *
   * This fix funtion writet the dynamic check and fix code to the excact dim
   * of the src dynamic dim to be same with the dest static dim
   */
  // Set environment
  mlir::Location loc = op->getLoc();
  mlir::OpBuilder builder(op->getBlock(), op->getBlock()->begin());
  builder.setInsertionPoint(op);

  // Get the Operand
  mlir::Value destTensor = op->getResult(0);
  mlir::ShapedType destTensorSTy =
      destTensor.getType().dyn_cast<mlir::ShapedType>();
  mlir::RankedTensorType destTensorTy =
      destTensor.getType().dyn_cast<mlir::RankedTensorType>();

  mlir::Value srcTensor = op->getOperand(0);
  mlir::ShapedType srcTensorSTy =
      srcTensor.getType().dyn_cast<mlir::ShapedType>();
  mlir::RankedTensorType srcTensorTy =
      srcTensor.getType().dyn_cast<mlir::RankedTensorType>();
  // // for(int64_t i = 0; i < dynDimsIdices.size(); i++) {
  // //   llvm::outs() << dynDimsIdices[i] << " ";
  // // }
  // // llvm::outs() << "\n";

  // for ever tocompare dim pairs, check the consistence, form the dynamic
  // attribute for tensor.splat
  llvm::SmallVector<mlir::Value> fixedDynDimsIdices;
  mlir::Value shapeSame = ConstantPool::getValue(1, builder.getI1Type());
  bool needToCompare = false;
  for (int64_t i = 0; i < srcTensorSTy.getRank(); i++) {
    // input dim is static
    if (srcTensorSTy.getDimSize(i) > 0)
      continue;
    mlir::Value curDim = ConstantPool::getValue(i, builder.getIndexType());
    mlir::Value curInputSize =
        builder.create<mlir::tensor::DimOp>(loc, srcTensor, curDim).getResult();
    // both input and output dim are dynamic
    if (destTensorSTy.getDimSize(i) < 0) {
      fixedDynDimsIdices.push_back(curInputSize);
      continue;
    }
    needToCompare = true;
    // use the num same with the static dim as the attribute of the new dyn dim
    mlir::Value curOutputSize = ConstantPool::getValue(
        destTensorSTy.getDimSize(i), builder.getIndexType());
    fixedDynDimsIdices.push_back(curOutputSize);

    mlir::Value cmpResult =
        builder
            .create<mlir::index::CmpOp>(loc, mlir::index::IndexCmpPredicate::EQ,
                                        curInputSize, curOutputSize)
            .getResult();
    shapeSame = builder.create<mlir::arith::AndIOp>(loc, cmpResult, shapeSame)
                    .getResult();
  }

  // for the case of having the target type, insert scf.if and replace the ori
  // value
  if (needToCompare) {
    // Write the dynamic check and fix code.
    auto sameBranch = [&](mlir::OpBuilder &b, mlir::Location loc) {
      b.create<mlir::scf::YieldOp>(loc, srcTensor);
    };
    auto notSameBranch = [&](mlir::OpBuilder &b, mlir::Location loc) {
      // generate new dynamic dim size to cast tensor
      //  mlir::Value newSrcRetVal =
      //        builder.create<mlir::memref::AllocOp>(loc, srcTensorTy,
      //        fixedDynDimsIdices).getResult();
      mlir::Value randnumVal = FixUBUtils::genRandomIntOrFloatValue(
          builder, loc, srcTensorTy.getElementType());
      mlir::Value newSrcRetVal = builder.create<mlir::tensor::SplatOp>(
          loc, randnumVal, srcTensorTy, fixedDynDimsIdices);
      b.create<mlir::scf::YieldOp>(loc, newSrcRetVal);
    };
    mlir::Operation *scfIfOp =
        builder
            .create<mlir::scf::IfOp>(loc, shapeSame, sameBranch, notSameBranch)
            .getOperation();
    op->setOperand(0, scfIfOp->getResult(0));
  }

  return;
}

void FixTensor::fixExtract(mlir::Operation *op) {
  /**
   * Fix undefined behaviors in tensor.extract.
   *
   * Example:
   * %4 = tensor.extract %t[%1, %2] : tensor<4x4xi32>
   *
   * This fix work does the following tasks:
   * (1) Ensure that the indices are not out-of-bounds
   */
  // Set environment
  mlir::Location loc = op->getLoc();
  mlir::OpBuilder builder(op->getBlock(), op->getBlock()->begin());
  builder.setInsertionPoint(op);

  // Get operands
  mlir::Value tensor = op->getOperand(0);
  mlir::ShapedType tensorShapedType =
      mlir::dyn_cast<mlir::ShapedType>(tensor.getType());
  llvm::SmallVector<mlir::Value> indices = llvm::SmallVector<mlir::Value>();
  for (int64_t i = 1; i < op->getNumOperands(); ++i)
    indices.push_back(op->getOperand(i));

  // Get tensor shape
  llvm::SmallVector<mlir::Value> tensorDims = llvm::SmallVector<mlir::Value>();
  FixUBUtils::getShapes(builder, loc, tensor, &tensorDims);

  // Generate new shape
  llvm::SmallVector<mlir::Value> newDims = llvm::SmallVector<mlir::Value>();
  FixUBUtils::getValidIndex(builder, loc, tensorDims, indices, &newDims);

  // Set new op
  for (int64_t i = 1; i < op->getNumOperands(); ++i) {
    op->setOperand(i, newDims[i - 1]);
  }

  return;
}

void FixTensor::fixInsert(mlir::Operation *op) {
  /**
   * Fix undefined behaviors in tensor.insert.
   *
   * Example:
   * %4 = tensor.insert %t into %dest[%1, %2] : tensor<4x4xi32>
   *
   * This fix work does the following tasks:
   * (1)
   */
  // Set environment
  mlir::Location loc = op->getLoc();
  mlir::OpBuilder builder(op->getBlock(), op->getBlock()->begin());
  builder.setInsertionPoint(op);

  // get the target tensor
  mlir::Value tensor = op->getOperand(1);
  llvm::SmallVector<mlir::Value> indices = llvm::SmallVector<mlir::Value>();
  for (int64_t i = 2; i < op->getNumOperands(); ++i) {
    indices.push_back(op->getOperand(i));
  }

  // Get shapes
  llvm::SmallVector<mlir::Value> tensorShape = llvm::SmallVector<mlir::Value>();
  FixUBUtils::getShapes(builder, loc, tensor, &tensorShape);

  // Generate new indices
  llvm::SmallVector<mlir::Value> newIndices = llvm::SmallVector<mlir::Value>();
  FixUBUtils::getValidIndex(builder, loc, tensorShape, indices, &newIndices);

  // Set operands
  for (int64_t i = 2; i < op->getNumOperands(); ++i) {
    op->setOperand(i, newIndices[i - 2]);
  }

  return;
}