#include "mlir/Dialect/Affine/Passes.h"
// #include "mlir/Dialect/Index/IR/IndexEnums.h.inc"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Pass/Pass.h"

#include "fixub/ConstantPool.h"
#include "fixub/DelOp.h"
#include "fixub/FixUBUtils.h"
#include "fixub/FixVector.h"

// Note that: LLVM backend may not support vector with more than 1 dimensions!

// Note that: MLIR does not support dynamic shaped vector!

// Note: all operations has been checked, they are all supported by MLIRSmith,
// and can be lowered to LLVM dialect!

// The driver function
void FixVector::fixvector(mlir::Operation *op) {
  // vector.compressstore - yes
  // vector.expandload - yes
  // vector.gather - yes
  // vector.maskedload - yes
  // vector.maskedstore - yes
  // vector.scatter - yes
  // vector.reshape - I am not sure, what the undefined elements will be...

  std::string opname = op->getName().getStringRef().str();
  if (opname == "vector.maskedload" || opname == "vector.maskedstore") {
    FixVector::fixMaskLoadAndStore(op);
  }
  if (opname == "vector.compressstore" || opname == "vector.expandload") {
    FixVector::fixCompressStoreAndExpandLoad(op);
  }
  if (opname == "vector.gather") {
    FixVector::fixgather(op);
  }
  if (opname == "vector.scatter") {
    FixVector::fixscatter(op);
  }
  if (opname == "vector.insertelement") {
    FixVector::fixInsertElement(op);
  }
  if (opname == "vector.load") {
    FixVector::fixLoad(op);
  }
  if (opname == "vector.reshape") {
  }
  return;
}

void FixVector::fixIndices(mlir::Operation *op) {
  /**
   * Fix the indices to be in range of the corresponding size of base.
   * Speficically, do the following steps:
   *
   * Params:
   * mlir::Operation* op, which is the operation to be fixed.
   * llvm::SmallVector<mlir::Value> indices, which is the vector of indices.
   * mlir::Value base, which is a memref type value.
   *
   * Returns:
   * None
   */
  // set the mutation environment
  mlir::Location loc = op->getLoc();
  mlir::OpBuilder builder(op->getBlock(), op->getBlock()->begin());
  builder.setInsertionPoint(op);

  // Read the operands
  int64_t operandsNum = op->getNumOperands();
  mlir::Value baseMem = op->getOperand(0);
  mlir::ShapedType baseMemSTy = baseMem.getType().dyn_cast<mlir::ShapedType>();
  int64_t baseMemRank = baseMemSTy.getRank();
  llvm::SmallVector<mlir::Value> indices = llvm::SmallVector<mlir::Value>();
  for (int64_t i = 0; i < baseMemRank; i++) {
    indices.push_back(op->getOperand(i + 1));
  }
  mlir::Value maskVec = op->getOperand(0 + baseMemRank + 1);
  mlir::Value valueVec = op->getOperand(0 + baseMemRank + 2);

  // Get the dimension size of base memref, and fix the indices according to the
  // got size.
  for (int64_t i = 0; i < baseMemRank; i++) {
    mlir::Value curIdx = indices[i];
    mlir::Value baseMemDimsizeVal;
    if (baseMemSTy.isDynamic(i) || baseMemSTy.getDimSize(i) <= 0) {
      // a dynamic fix.
      mlir::Value curDim = ConstantPool::getValue(i, builder.getIndexType());
      baseMemDimsizeVal =
          builder.create<mlir::memref::DimOp>(loc, baseMem, curDim).getResult();
    } else {
      // a static fix.
      int64_t baseMemDimsize = baseMemSTy.getDimSize(i);
      baseMemDimsizeVal =
          ConstantPool::getValue(baseMemDimsize, builder.getIndexType());
    }
    // calculate the new index
    mlir::Value newIdx;
    if (i == baseMemRank - 1) {
      if (i == baseMemRank - 1) {
        baseMemDimsizeVal =
            builder
                .create<mlir::index::SubOp>(
                    loc, baseMemDimsizeVal,
                    ConstantPool::getValue(1, builder.getIndexType()))
                .getResult();
      }
      mlir::Value zeroDimRequest =
          builder
              .create<mlir::index::CmpOp>(
                  op->getLoc(), mlir::index::IndexCmpPredicate::EQ,
                  ConstantPool::getValue(0, builder.getIndexType()),
                  baseMemDimsizeVal)
              .getResult();
      auto zeroBranchBuilder = [&](mlir::OpBuilder &b, mlir::Location loc) {
        b.create<mlir::scf::YieldOp>(
            loc, ConstantPool::getValue(0, builder.getIndexType()));
      };
      auto nonZeroBranchBuilder = [&](mlir::OpBuilder &b, mlir::Location loc) {
        mlir::Value newIdx =
            builder.create<mlir::index::RemUOp>(loc, curIdx, baseMemDimsizeVal)
                .getResult();
        b.create<mlir::scf::YieldOp>(loc, newIdx);
      };
      newIdx =
          builder
              .create<mlir::scf::IfOp>(loc, zeroDimRequest, zeroBranchBuilder,
                                       nonZeroBranchBuilder)
              .getOperation()
              ->getResult(0);
    } else {
      newIdx =
          builder.create<mlir::index::RemUOp>(loc, curIdx, baseMemDimsizeVal)
              .getResult();
    }
    op->setOperand(1 + i, newIdx);
  }
}

mlir::Operation *FixVector::getLastDimSize(mlir::Value memref) {
  /**
   * Get the last dimension size of a given memref.
   *
   * Param:
   * mlir::Value memref, which is the calculated memref.
   *
   * Return:
   * The pointer of operation that returns the dimension size.
   */
  // Set the environment.
  mlir::Operation *op = memref.getDefiningOp();
  mlir::Location loc = op->getLoc();
  mlir::OpBuilder builder(op->getBlock(), op->getBlock()->begin());
  builder.setInsertionPointAfter(op);

  // Calculate the required information.
  mlir::ShapedType baseMemSTy = memref.getType().dyn_cast<mlir::ShapedType>();
  int64_t baseMemRank = baseMemSTy.getRank();

  mlir::Operation *baseMemLastDimSizeOp;
  if (baseMemSTy.isDynamic(baseMemRank - 1) ||
      baseMemSTy.getDimSize(baseMemRank - 1) <= 0) {
    // the dynamic branch.
    mlir::Value curDim =
        ConstantPool::getValue(baseMemRank - 1, builder.getIndexType());
    baseMemLastDimSizeOp =
        builder.create<mlir::memref::DimOp>(loc, memref, curDim).getOperation();
  } else {
    // the static branch.
    int64_t baseMemDimsize = baseMemSTy.getDimSize(baseMemRank - 1);
    baseMemLastDimSizeOp =
        ConstantPool::getValue(baseMemDimsize, builder.getIndexType())
            .getDefiningOp();
  }
  return baseMemLastDimSizeOp;
}

// vector.compressstore - yes
void FixVector::fixMaskLoadAndStore(mlir::Operation *op) {
  /***
   * The fix of opeartion: vector.compressstore, the fix steps are:
   * (1) Fix the indices by directly use a%b.
   * (2) Fix the content of mask to be 0, if the index is out-of-bound.
   */
  // Set the fix environment
  mlir::Location loc = op->getLoc();
  mlir::OpBuilder builder(op->getBlock(), op->getBlock()->begin());
  builder.setInsertionPoint(op);

  // Fix the indices
  FixVector::fixIndices(op);

  // Re-load the operands
  int64_t operandsNum = op->getNumOperands();
  mlir::Value baseMem = op->getOperand(0);
  mlir::ShapedType baseMemSTy = baseMem.getType().dyn_cast<mlir::ShapedType>();
  int64_t baseMemRank = baseMemSTy.getRank();
  llvm::SmallVector<mlir::Value> indices = llvm::SmallVector<mlir::Value>();
  for (int64_t i = 0; i < baseMemRank; i++) {
    indices.push_back(op->getOperand(i + 1));
  }
  mlir::Value maskVec = op->getOperand(0 + baseMemRank + 1);
  mlir::Value valueVec = op->getOperand(0 + baseMemRank + 2);

  // Calcualte required information, i.e. the size of last dimension
  mlir::Value baseMemLastDimSize = getLastDimSize(baseMem)->getResult(0);

  // Fix the mask
  mlir::Value lb = ConstantPool::getValue(0, builder.getIndexType());
  mlir::Value step = ConstantPool::getValue(1, builder.getIndexType());
  mlir::ShapedType maskTy = maskVec.getType().dyn_cast<mlir::ShapedType>();
  int64_t maskSize = maskTy.getDimSize(0);
  mlir::Value ub = ConstantPool::getValue(maskSize, builder.getIndexType());
  llvm::SmallVector<mlir::Value> iterArgs = llvm::SmallVector<mlir::Value>();
  iterArgs.push_back(maskVec);
  auto blockBuilder = [&](mlir::OpBuilder &builder, mlir::Location loc,
                          mlir::Value iv /*loop iterator*/,
                          mlir::ValueRange args) {
    mlir::Value mask = args[0];
    mlir::Value curIdx = builder.create<mlir::index::AddOp>(
        loc, indices[indices.size() - 1], iv);
    // compare the current index and size of last dimension of base memref, and
    // fix the invalid mask.
    mlir::Value valid = builder
                            .create<mlir::index::CmpOp>(
                                loc, mlir::index::IndexCmpPredicate::UGT,
                                baseMemLastDimSize, curIdx)
                            .getResult();
    auto trueBuilder = [&](mlir::OpBuilder &b, mlir::Location loc) {
      builder.create<mlir::scf::YieldOp>(loc, mask);
    };
    auto falseBuilder = [&](mlir::OpBuilder &b, mlir::Location loc) {
      mlir::Value newMask = builder.create<mlir::vector::InsertElementOp>(
          loc, ConstantPool::getValue(0, builder.getI1Type()), mask, iv);
      builder.create<mlir::scf::YieldOp>(loc, newMask);
    };
    mlir::Operation *scfIfOp =
        builder.create<mlir::scf::IfOp>(loc, valid, trueBuilder, falseBuilder)
            .getOperation();
    builder.create<mlir::scf::YieldOp>(loc, scfIfOp->getResults());
  };
  mlir::Operation *scfForOp = builder.create<mlir::scf::ForOp>(
      loc, lb, ub, step, iterArgs, blockBuilder);
  op->setOperand(operandsNum - 2, scfForOp->getResult(0));
  return;
}

void FixVector::fixCompressStoreAndExpandLoad(mlir::Operation *op) {
  // Set the environment
  mlir::Location loc = op->getLoc();
  mlir::OpBuilder builder(op->getBlock(), op->getBlock()->begin());
  builder.setInsertionPoint(op);

  // Fix the indices
  FixVector::fixIndices(op);

  // Re-load the operands
  int64_t operandsNum = op->getNumOperands();
  mlir::Value baseMem = op->getOperand(0);
  mlir::ShapedType baseMemSTy = baseMem.getType().dyn_cast<mlir::ShapedType>();
  int64_t baseMemRank = baseMemSTy.getRank();
  llvm::SmallVector<mlir::Value> indices = llvm::SmallVector<mlir::Value>();
  for (int64_t i = 0; i < baseMemRank; i++) {
    indices.push_back(op->getOperand(i + 1));
  }
  mlir::Value maskVec = op->getOperand(0 + baseMemRank + 1);
  mlir::Value valueVec = op->getOperand(0 + baseMemRank + 2);

  // Calcualte required information, i.e. the size of last dimension
  mlir::Value baseMemLastDimSize = getLastDimSize(baseMem)->getResult(0);

  // Fix the mask,
  // different from vector.maskedload and vector.maksedstore,
  // the fix of these two operation compares the summary of mask and the index.

  // Prepare for lower bound, upper bound, step, and iteration arguments in
  // scf.for
  mlir::Value lb = ConstantPool::getValue(0, builder.getIndexType());
  mlir::Value step = ConstantPool::getValue(1, builder.getIndexType());
  mlir::ShapedType maskTy = maskVec.getType().dyn_cast<mlir::ShapedType>();
  int64_t maskSize = maskTy.getDimSize(0);
  mlir::Value ub = ConstantPool::getValue(maskSize, builder.getIndexType());
  llvm::SmallVector<mlir::Value> iterArgs = llvm::SmallVector<mlir::Value>();
  iterArgs.push_back(maskVec);
  iterArgs.push_back(ConstantPool::getValue(0, builder.getI32Type()));
  auto blockBuilder = [&](mlir::OpBuilder &builder, mlir::Location loc,
                          mlir::Value iv /*loop iterator*/,
                          mlir::ValueRange args) {
    mlir::Value mask = args[0];
    mlir::Value maskSum = args[1];

    // Calculate the offset
    mlir::Value maskElem =
        builder.create<mlir::vector::ExtractElementOp>(loc, mask, iv)
            .getOperation()
            ->getResult(0);
    auto maskTrueBuilder = [&](mlir::OpBuilder &b, mlir::Location loc) {
      mlir::Value newMaskSum =
          builder
              .create<mlir::arith::AddIOp>(
                  loc, maskSum, ConstantPool::getValue(1, builder.getI32Type()))
              .getOperation()
              ->getResult(0);
      builder.create<mlir::scf::YieldOp>(loc, newMaskSum);
    };
    auto maskFalseBuilder = [&](mlir::OpBuilder &b, mlir::Location loc) {
      builder.create<mlir::scf::YieldOp>(loc, maskSum);
    };
    mlir::Operation *maskScfIfOp =
        builder
            .create<mlir::scf::IfOp>(loc, maskElem, maskTrueBuilder,
                                     maskFalseBuilder)
            .getOperation();
    maskSum = maskScfIfOp->getResult(0);

    // Calculate the current index
    mlir::Value maskSumIdx =
        builder
            .create<mlir::index::CastSOp>(loc, builder.getIndexType(), maskSum)
            .getOperation()
            ->getResult(0);
    mlir::Value curIdx = builder.create<mlir::index::AddOp>(
        loc, indices[indices.size() - 1], maskSumIdx);

    // compare the current index and size of last dimension of base memref, and
    // fix the invalid mask.
    mlir::Value valid = builder
                            .create<mlir::index::CmpOp>(
                                loc, mlir::index::IndexCmpPredicate::UGT,
                                baseMemLastDimSize, curIdx)
                            .getResult();
    auto trueBuilder = [&](mlir::OpBuilder &b, mlir::Location loc) {
      builder.create<mlir::scf::YieldOp>(loc, mask);
    };
    auto falseBuilder = [&](mlir::OpBuilder &b, mlir::Location loc) {
      mlir::Value newMask = builder.create<mlir::vector::InsertElementOp>(
          loc, ConstantPool::getValue(0, builder.getI1Type()), mask, iv);
      builder.create<mlir::scf::YieldOp>(loc, newMask);
    };
    mlir::Operation *scfIfOp =
        builder.create<mlir::scf::IfOp>(loc, valid, trueBuilder, falseBuilder)
            .getOperation();

    mlir::Value newMask = scfIfOp->getResult(0);
    llvm::SmallVector<mlir::Value> returnVals =
        llvm::SmallVector<mlir::Value>();
    returnVals.push_back(newMask);
    returnVals.push_back(maskSum);
    builder.create<mlir::scf::YieldOp>(loc, returnVals);
  };
  mlir::Operation *scfForOp = builder.create<mlir::scf::ForOp>(
      loc, lb, ub, step, iterArgs, blockBuilder);
  op->setOperand(operandsNum - 2, scfForOp->getResult(0));
  return;
}

void FixVector::fixgather(mlir::Operation *op) {
  /***
   * Fix the undefined behaviors in vector.gather.
   * The documentation is:
   * If a mask bit is set and the corresponding index is out-of-bounds for the
   * given base, the behavior is undefined. If a mask bit is not set, the value
   * comes from the pass-through vector regardless of the index, and the index
   * is allowed to be out-of-bounds. The mainly fix task is: (1) Fix all indices
   * be in range of corresponding dimension of base. (2) Check indices of mask
   * is out-of-bounds, if it is, set the mask to 0, otherwise, remain.
   */
  // Set the environment
  mlir::Location loc = op->getLoc();
  mlir::OpBuilder builder(op->getBlock(), op->getBlock()->begin());
  builder.setInsertionPoint(op);

  // Fix the indices
  FixVector::fixIndices(op);

  // Reload the parameters
  int64_t operandsNum = op->getNumOperands();
  mlir::Value passThru = op->getOperand(operandsNum - 1);
  mlir::Value mask = op->getOperand(operandsNum - 2);
  mlir::Value indexVec = op->getOperand(operandsNum - 3);
  mlir::Value base = op->getOperand(0);
  llvm::SmallVector<mlir::Value> indices = llvm::SmallVector<mlir::Value>();
  for (int64_t i = 1; i < operandsNum - 3; i++) {
    indices.push_back(op->getOperand(i));
  }
  // addtional informations
  mlir::ShapedType baseSTy = base.getType().dyn_cast<mlir::ShapedType>();
  int64_t baseRank = baseSTy.getRank();

  // Calcualte required information, i.e. the size of last dimension
  mlir::Value baseLastDimSize = getLastDimSize(base)->getResult(0);

  // Fix masks
  // 1. Convert mask and indexVec to 1-dim vector.
  // 1.1 Get old shape and new shape.
  mlir::ShapedType maskSTy = mask.getType().dyn_cast<mlir::ShapedType>();
  auto oldShape =
      maskSTy.getShape(); // assumption: all vector should be static shaped.
  mlir::ShapedType indexVecSTy =
      indexVec.getType().dyn_cast<mlir::ShapedType>();
  int64_t maskNumElements = 1;
  for (int64_t dimSize : oldShape) {
    maskNumElements *= dimSize;
  }
  // 1.2 generate the target shape.
  llvm::SmallVector<int64_t> newShape = llvm::SmallVector<int64_t>();
  newShape.push_back(maskNumElements);
  mlir::VectorType newMaskVecTy =
      mlir::VectorType::get(newShape, maskSTy.getElementType());
  mlir::VectorType newIndexVecTy =
      mlir::VectorType::get(newShape, indexVecSTy.getElementType());
  mlir::Value newMask =
      builder.create<mlir::vector::ShapeCastOp>(loc, newMaskVecTy, mask);
  mlir::Value newIndexVec =
      builder.create<mlir::vector::ShapeCastOp>(loc, newIndexVecTy, indexVec);
  // 2. Traverse the mask and indexVec, fix the value of mask
  // 2.1 Calculate the upper bound, lower bound, step, and iteration arguments
  // for scf.for
  mlir::Value ub =
      ConstantPool::getValue(maskNumElements, builder.getIndexType());
  mlir::Value lb = ConstantPool::getValue(0, builder.getIndexType());
  mlir::Value step = ConstantPool::getValue(1, builder.getIndexType());
  llvm::SmallVector<mlir::Value> iterArgs = llvm::SmallVector<mlir::Value>();
  iterArgs.push_back(newMask);
  // 2.2 Create the scf.for, and write the check logic
  auto blockBuilder = [&](mlir::OpBuilder &builder, mlir::Location loc,
                          mlir::Value iv /*loop iterator*/,
                          mlir::ValueRange args) {
    mlir::Value mask = args[0];
    // extract value of indexVec
    mlir::Value indexI32 =
        builder.create<mlir::vector::ExtractElementOp>(loc, newIndexVec, iv)
            .getOperation()
            ->getResult(0); // i32
    mlir::Value index =
        builder
            .create<mlir::index::CastSOp>(loc, builder.getIndexType(), indexI32)
            .getOperation()
            ->getResult(0);
    mlir::Value curIdx = builder.create<mlir::index::AddOp>(loc, index, iv)
                             .getOperation()
                             ->getResult(0);
    // compare the current index and size of last dimension of base memref, and
    // fix the invalid mask.
    mlir::Value valid = builder
                            .create<mlir::index::CmpOp>(
                                loc, mlir::index::IndexCmpPredicate::UGT,
                                baseLastDimSize, curIdx)
                            .getOperation()
                            ->getResult(0);
    auto trueBuilder = [&](mlir::OpBuilder &b, mlir::Location loc) {
      builder.create<mlir::scf::YieldOp>(loc, mask);
    };
    auto falseBuilder = [&](mlir::OpBuilder &b, mlir::Location loc) {
      mlir::Value newMask = builder.create<mlir::vector::InsertElementOp>(
          loc, ConstantPool::getValue(0, builder.getI1Type()), mask, iv);
      builder.create<mlir::scf::YieldOp>(loc, newMask);
    };
    mlir::Operation *scfIfOp =
        builder.create<mlir::scf::IfOp>(loc, valid, trueBuilder, falseBuilder)
            .getOperation();
    builder.create<mlir::scf::YieldOp>(loc, scfIfOp->getResults());
  };
  mlir::Operation *scfForOp = builder.create<mlir::scf::ForOp>(
      loc, lb, ub, step, iterArgs, blockBuilder);
  newMask = scfForOp->getResult(0);
  // 3. Convert mask and indexVec to original shape.
  mlir::Value fixedMask = builder.create<mlir::vector::ShapeCastOp>(
      loc, mask.getType().dyn_cast<mlir::VectorType>(), newMask);
  op->setOperand(operandsNum - 2, fixedMask);
  return;
}

void FixVector::fixscatter(mlir::Operation *op) {
  /**
   * Fix undefined behaviors in vector.scatter operation. Example:
   *
   * vector.scatter %base[%index, %index][%index_vec], %mask, %value :
   * memref<?x?xf32>, vector<16x16xi32>, vector<16x16xi1>, vector<16x16xf32>
   *
   * Description:
   * If a mask bit is set and the corresponding index is out-of-bounds for the
   * given base, the behavior is undefined. If a mask bit is not set, no value
   * is stored regardless of the index, and the index is allowed to be
   * out-of-bounds.
   *
   * If the index vector contains two or more duplicate indices, the behavior is
   * undefined. Underlying implementation may enforce strict sequential
   * semantics. TODO: always enforce strict sequential semantics?
   *
   * (1) Make the %index_vec unique. Because we do not know whether duplicate
   * indices are allowed when corresponding masks are set to 0, we ensure it's
   * uniqueness first. (2) Fix the mask, as FixVector::fixgather do.
   */

  // Set the environment.
  mlir::Location loc = op->getLoc();
  mlir::OpBuilder builder(op->getBlock(), op->getBlock()->begin());
  builder.setInsertionPoint(op);

  // Load the operands
  mlir::Value base = op->getOperand(0);
  int64_t operandsNum = op->getNumOperands();
  mlir::Value value = op->getOperand(operandsNum - 1);
  mlir::Value mask = op->getOperand(operandsNum - 2);
  mlir::Value indexVec = op->getOperand(operandsNum - 3);
  llvm::SmallVector<mlir::Value> indices = llvm::SmallVector<mlir::Value>();
  for (int64_t i = 1; i < operandsNum - 3; i++) {
    indices.push_back(op->getOperand(i));
  }

  // Calculate required information
  mlir::ShapedType indexVecSTy =
      indexVec.getType().dyn_cast<mlir::ShapedType>();
  llvm::ArrayRef<int64_t> indexVecShape = indexVecSTy.getShape();
  int64_t indexVecSize =
      indexVecShape[0]; // Note: this operation only accept 1-dim vector.

  // 1: Fix indexVec, i.e., make it's elements unique.
  // 1.1 Create a mask table that records whether with-in-bounds values are
  // used.
  mlir::VectorType inRangeMaskTy =
      mlir::VectorType::get(indexVecShape, builder.getI1Type());
  mlir::Value inRangeMask = builder.create<mlir::vector::SplatOp>(
      loc, ConstantPool::getValue(0, builder.getI1Type()), inRangeMaskTy);
  // 1.2 Create a table that records all used values out-of-bounds.
  mlir::Value outRangeTable = builder.create<mlir::vector::SplatOp>(
      loc, ConstantPool::getValue(-1, builder.getI32Type()),
      indexVec.getType().dyn_cast<mlir::VectorType>());
  // 1.3 Traverse the indexVec
  // 1.3.1 Prepare the lower bound, upper bound, steps, iteration arguments
  mlir::Value lb = ConstantPool::getValue(0, builder.getIndexType());
  mlir::Value ub = ConstantPool::getValue(indexVecSize, builder.getIndexType());
  mlir::Value step = ConstantPool::getValue(1, builder.getIndexType());
  llvm::SmallVector<mlir::Value> iterArgs = llvm::SmallVector<mlir::Value>();
  iterArgs.push_back(indexVec);
  iterArgs.push_back(inRangeMask);
  iterArgs.push_back(outRangeTable);
  iterArgs.push_back(
      ConstantPool::getValue(0, builder.getIndexType())); // outRangeTableIdx
  // 1.3.2 Create scf for loop`
  auto blockBuilder = [&](mlir::OpBuilder &builder, mlir::Location loc,
                          mlir::Value iv /*loop iterator*/,
                          mlir::ValueRange args) {
    // 1.3.2.1 Extract the element, check whether it has been used.
    mlir::Value indexVec = args[0];
    mlir::Value inRangeMask = args[1];
    mlir::Value outRangeTable = args[2];
    mlir::Value outRangeTableIdx = args[3];

    // 1.3.2.2 Check whether indexVecElem is smaller than zero.
    // Repalce "a" with "-(a+1)" when "a" is negative. Note that, "+1" is used
    // to prevent singed integer overflow.
    mlir::Value indexVecElem =
        builder.create<mlir::vector::ExtractElementOp>(loc, indexVec, iv)
            .getOperation()
            ->getResult(0);
    mlir::Value negIdx =
        builder
            .create<mlir::arith::CmpIOp>(
                loc, mlir::arith::CmpIPredicate::sgt,
                ConstantPool::getValue(0, builder.getI32Type()), indexVecElem)
            .getOperation()
            ->getResult(0);
    auto negBranch = [&](mlir::OpBuilder &b, mlir::Location loc) {
      mlir::Value newIndexVecElem =
          builder
              .create<mlir::arith::AddIOp>(
                  loc, indexVecElem,
                  ConstantPool::getValue(1, builder.getI32Type()))
              .getOperation()
              ->getResult(0); // add 1 to prevent signed integer overflow
      newIndexVecElem =
          builder
              .create<mlir::arith::MulIOp>(
                  loc, newIndexVecElem,
                  ConstantPool::getValue(-1, builder.getI32Type()))
              .getOperation()
              ->getResult(0);
      mlir::Value newIndexVec = builder.create<mlir::vector::InsertElementOp>(
          loc, newIndexVecElem, indexVec, iv);
      b.create<mlir::scf::YieldOp>(loc, newIndexVec);
    };
    auto nonNegBranch = [&](mlir::OpBuilder &b, mlir::Location loc) {
      b.create<mlir::scf::YieldOp>(loc, indexVec);
    };
    indexVec =
        builder.create<mlir::scf::IfOp>(loc, negIdx, negBranch, nonNegBranch)
            .getOperation()
            ->getResult(0);

    // 1.3.2.3 Check whether indexVecElem is used or not.
    indexVecElem =
        builder.create<mlir::vector::ExtractElementOp>(loc, indexVec, iv)
            .getOperation()
            ->getResult(0); // reload
    mlir::Value indexVecElemIndex =
        builder
            .create<mlir::index::CastSOp>(loc, builder.getIndexType(),
                                          indexVecElem)
            .getOperation()
            ->getResult(0);
    mlir::Value inRange =
        builder
            .create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::sgt,
                                         ub, indexVecElemIndex)
            .getOperation()
            ->getResult(0);
    mlir::Value notInRange =
        builder
            .create<mlir::arith::CmpIOp>(
                loc, mlir::arith::CmpIPredicate::ne, inRange,
                ConstantPool::getValue(1, builder.getI1Type()))
            .getOperation()
            ->getResult(0);
    mlir::Value isUsed = ConstantPool::getValue(0, builder.getI1Type());
    auto inRangeBranch = [&](mlir::OpBuilder &b, mlir::Location loc) {
      // Extract the corresponding element of inRangeMask, and check whether
      // indexVecElem is used.
      mlir::Value newIsUsed = builder
                                  .create<mlir::vector::ExtractElementOp>(
                                      loc, inRangeMask, indexVecElem)
                                  .getOperation()
                                  ->getResult(0);
      b.create<mlir::scf::YieldOp>(loc, newIsUsed);
    };
    auto nonInRangeBranch = [&](mlir::OpBuilder &b, mlir::Location loc) {
      // Traverse the outRangeTable, and check whether the value has been used.
      // Prepare the lower bound, upper bound, steps, iteration arguments
      mlir::Value lb = ConstantPool::getValue(0, builder.getIndexType());
      mlir::Value ub = outRangeTableIdx;
      mlir::Value step = ConstantPool::getValue(1, builder.getIndexType());
      llvm::SmallVector<mlir::Value> iterArgs =
          llvm::SmallVector<mlir::Value>();
      iterArgs.push_back(isUsed);
      auto blockBuilder = [&](mlir::OpBuilder &builder, mlir::Location loc,
                              mlir::Value iv /*loop iterator*/,
                              mlir::ValueRange args) {
        mlir::Value isUsed = args[0];
        // Extract the element content, and check whether it is equils with the
        // given element.
        mlir::Value usedValue =
            builder
                .create<mlir::vector::ExtractElementOp>(loc, outRangeTable, iv)
                .getOperation()
                ->getResult(0);
        mlir::Value curUsed = builder
                                  .create<mlir::arith::CmpIOp>(
                                      loc, mlir::arith::CmpIPredicate::eq,
                                      usedValue, indexVecElem)
                                  .getOperation()
                                  ->getResult(0);
        isUsed = builder.create<mlir::arith::AndIOp>(loc, curUsed, isUsed)
                     .getOperation()
                     ->getResult(0);
        b.create<mlir::scf::YieldOp>(loc, isUsed);
      };
      mlir::Operation *scfForOp =
          builder
              .create<mlir::scf::ForOp>(loc, lb, ub, step, iterArgs,
                                        blockBuilder)
              .getOperation();
      isUsed = scfForOp->getResult(0);
      b.create<mlir::scf::YieldOp>(loc, isUsed);
    };
    isUsed = builder
                 .create<mlir::scf::IfOp>(loc, inRange, inRangeBranch,
                                          nonInRangeBranch)
                 .getOperation()
                 ->getResult(0);
    mlir::Value notUsed =
        builder
            .create<mlir::arith::CmpIOp>(
                loc, mlir::arith::CmpIPredicate::ne, isUsed,
                ConstantPool::getValue(1, builder.getI1Type()))
            .getOperation()
            ->getResult(0);

    // If it is used, find the un-used index, and mark it as used.
    auto trueBranch = [&](mlir::OpBuilder &b, mlir::Location loc) {
      // Traverse inRangeMask, find the un-used ones.
      // Prepare the lower bound, upper bound, steps, iteration arguments
      mlir::Value lastNotUsed =
          ConstantPool::getValue(-1, builder.getIndexType());
      mlir::Value lb = ConstantPool::getValue(0, builder.getIndexType());
      mlir::Value ub =
          ConstantPool::getValue(indexVecSize, builder.getIndexType());
      mlir::Value step = ConstantPool::getValue(1, builder.getIndexType());
      llvm::SmallVector<mlir::Value> iterArgs =
          llvm::SmallVector<mlir::Value>();
      iterArgs.push_back(lastNotUsed);
      auto blockBuilder = [&](mlir::OpBuilder &builder, mlir::Location loc,
                              mlir::Value iv /*loop iterator*/,
                              mlir::ValueRange args) {
        mlir::Value lastNotUsed = args[0];
        mlir::Value isUsed =
            builder.create<mlir::vector::ExtractElementOp>(loc, inRangeMask, iv)
                .getOperation()
                ->getResult(0);
        auto isUsedBranch = [&](mlir::OpBuilder &b, mlir::Location loc) {
          b.create<mlir::scf::YieldOp>(loc, lastNotUsed);
        };
        auto notUsedBranch = [&](mlir::OpBuilder &b, mlir::Location loc) {
          b.create<mlir::scf::YieldOp>(loc, iv);
        };
        lastNotUsed = builder
                          .create<mlir::scf::IfOp>(loc, isUsed, isUsedBranch,
                                                   notUsedBranch)
                          .getOperation()
                          ->getResult(0);
        b.create<mlir::scf::YieldOp>(loc, lastNotUsed);
      };
      mlir::Operation *scfForOp =
          builder
              .create<mlir::scf::ForOp>(loc, lb, ub, step, iterArgs,
                                        blockBuilder)
              .getOperation();
      // Get the un-used element
      mlir::Value unUsedIdx = scfForOp->getResult(0);
      // Set this position to the unused value, and set the corresponding mask
      // to be 1
      mlir::Value unUsedIdxI32 = builder
                                     .create<mlir::index::CastSOp>(
                                         loc, builder.getI32Type(), unUsedIdx)
                                     .getOperation()
                                     ->getResult(0);
      mlir::Value newIndexVec = builder
                                    .create<mlir::vector::InsertElementOp>(
                                        loc, unUsedIdxI32, indexVec, iv)
                                    .getOperation()
                                    ->getResult(0);
      mlir::Value newInRangeMask =
          builder
              .create<mlir::vector::InsertElementOp>(
                  loc, ConstantPool::getValue(1, builder.getI1Type()),
                  inRangeMask, unUsedIdx)
              .getOperation()
              ->getResult(0);
      llvm::SmallVector<mlir::Value> scfRetVals =
          llvm::SmallVector<mlir::Value>();
      scfRetVals.push_back(newIndexVec);
      scfRetVals.push_back(newInRangeMask);
      b.create<mlir::scf::YieldOp>(loc, scfRetVals);
    };
    auto falseBranch = [&](mlir::OpBuilder &b, mlir::Location loc) {
      llvm::SmallVector<mlir::Value> scfRetVals =
          llvm::SmallVector<mlir::Value>();
      scfRetVals.push_back(indexVec);
      scfRetVals.push_back(inRangeMask);
      b.create<mlir::scf::YieldOp>(loc, scfRetVals);
    };
    mlir::Operation *scfIfOp =
        builder.create<mlir::scf::IfOp>(loc, isUsed, trueBranch, falseBranch)
            .getOperation();
    indexVec = scfIfOp->getResult(0);
    inRangeMask = scfIfOp->getResult(1);
    indexVecElem =
        builder.create<mlir::vector::ExtractElementOp>(loc, indexVec, iv)
            .getOperation()
            ->getResult(0); // reload

    // If it is not used, and in range; Set the corresponding bit to be 1.
    mlir::Value cond =
        builder.create<mlir::arith::AndIOp>(loc, notUsed, inRange)
            .getOperation()
            ->getResult(0);
    auto trueBranch1 = [&](mlir::OpBuilder &b, mlir::Location loc) {
      mlir::Value newInRangeMask =
          builder
              .create<mlir::vector::InsertElementOp>(
                  loc, ConstantPool::getValue(1, builder.getI1Type()),
                  inRangeMask, indexVecElem)
              .getOperation()
              ->getResult(0);
      b.create<mlir::scf::YieldOp>(loc, newInRangeMask);
    };
    auto falseBranch1 = [&](mlir::OpBuilder &b, mlir::Location loc) {
      b.create<mlir::scf::YieldOp>(loc, inRangeMask);
    };
    scfIfOp =
        builder.create<mlir::scf::IfOp>(loc, cond, trueBranch1, falseBranch1)
            .getOperation();
    inRangeMask = scfIfOp->getResult(0);

    // If it is not used, and out-of-range; Add the indexVecElem value into out
    // of range table, and it's index + 1
    cond = builder.create<mlir::arith::AndIOp>(loc, notUsed, notInRange)
               .getOperation()
               ->getResult(0);
    auto trueBranch2 = [&](mlir::OpBuilder &b, mlir::Location loc) {
      // add the unsed element into out-of-range vec
      mlir::Value newOutRangeTable =
          builder.create<mlir::vector::InsertElementOp>(
              loc, indexVecElem, outRangeTable, outRangeTableIdx);
      mlir::Value newOutRangeTableIdx = builder.create<mlir::index::AddOp>(
          loc, outRangeTableIdx,
          ConstantPool::getValue(1, builder.getIndexType()));
      llvm::SmallVector<mlir::Value> retVal = llvm::SmallVector<mlir::Value>();
      retVal.push_back(newOutRangeTable);
      retVal.push_back(newOutRangeTableIdx);
      b.create<mlir::scf::YieldOp>(loc, retVal);
    };
    auto falseBranch2 = [&](mlir::OpBuilder &b, mlir::Location loc) {
      llvm::SmallVector<mlir::Value> retVal = llvm::SmallVector<mlir::Value>();
      retVal.push_back(outRangeTable);
      retVal.push_back(outRangeTableIdx);
      b.create<mlir::scf::YieldOp>(loc, retVal);
    };
    scfIfOp =
        builder.create<mlir::scf::IfOp>(loc, cond, trueBranch2, falseBranch2)
            .getOperation();
    outRangeTable = scfIfOp->getResult(0);
    outRangeTableIdx = scfIfOp->getResult(1);

    llvm::SmallVector<mlir::Value> retVal = llvm::SmallVector<mlir::Value>();
    retVal.push_back(indexVec);
    retVal.push_back(inRangeMask);
    retVal.push_back(outRangeTable);
    retVal.push_back(outRangeTableIdx);
    builder.create<mlir::scf::YieldOp>(loc, retVal);
  };
  mlir::Operation *scfForOp = builder.create<mlir::scf::ForOp>(
      loc, lb, ub, step, iterArgs, blockBuilder);
  indexVec = scfForOp->getResult(0);
  inRangeMask = scfForOp->getResult(1);
  outRangeTable = scfForOp->getResult(2);
  mlir::Value outRangeTableIdx = scfForOp->getResult(3);
  op->setOperand(operandsNum - 3, indexVec);

  // 2 Fix the mask, directly reuse the fix logic of vector.gather
  FixVector::fixgather(op);
  return;
}

void FixVector::fixInsertElement(mlir::Operation *op) {
  /**
   * Fix undefined behaviors in vector.insertelement.
   *
   * Example:
   * %1 = vector.insertelement %c15_i32, %0[%idx25 : index] : vector<17xi32>
   *
   * This fix work does the following tasks:
   * (1) Fix the index out-of-bound error.
   *
   *
   * This fix funtion writet set the index as ori index mod vector dim size
   */
  // Set environment
  mlir::Location loc = op->getLoc();
  mlir::OpBuilder builder(op->getBlock(), op->getBlock()->begin());
  builder.setInsertionPoint(op);

  // Get the Operand
  mlir::ShapedType vectorSTy =
      op->getOperand(1).getType().dyn_cast<mlir::ShapedType>();
  if (vectorSTy.getRank() < 1)
    return;
  mlir::Value insertIdx = op->getOperand(2);
  // llvm::outs() << op->getOperand(1) << "\n";
  // llvm::outs() << vectorSTy << "\n";

  mlir::Value vectorDimVal =
      ConstantPool::getValue(vectorSTy.getDimSize(0), builder.getIndexType());

  // Cal index mod dim, and set the result as the new index
  mlir::Value remuVal =
      builder.create<mlir::index::RemUOp>(loc, insertIdx, vectorDimVal)
          .getResult();

  op->setOperand(2, remuVal);

  return;
}

void FixVector::fixLoad(mlir::Operation *op) {
  /**
   * Fix undefined behaviors in vector.load.
   *
   * Example:
   * %10 = vector.load %alloc[%c25] : memref<?xi16>, vector<21xi16>
   *
   * This fix work does the following tasks:
   * (1) Fix the index out-of-bound error.
   * (2) Fix the vector extract space out-of-bound error
   *
   *
   * This fix funtion writet set the index as ori index mod vector dim size
   * and generate new memref with enough space to extract
   */
  // Set the fix environment
  mlir::Location loc = op->getLoc();
  mlir::OpBuilder builder(op->getBlock(), op->getBlock()->begin());
  builder.setInsertionPoint(op);

  // Get the Operand
  mlir::Value baseMem = op->getOperand(0);
  mlir::ShapedType baseMemSTy = baseMem.getType().dyn_cast<mlir::ShapedType>();
  mlir::MemRefType baseMemTy = baseMem.getType().dyn_cast<mlir::MemRefType>();
  mlir::Value insertIdx = op->getOperand(1);
  // llvm::outs() << baseMem << " " << insertIdx << "\n";
  mlir::ShapedType vectorSTy =
      op->getResult(0).getType().dyn_cast<mlir::ShapedType>();
  mlir::Value vectorDimVal =
      ConstantPool::getValue(vectorSTy.getDimSize(0), builder.getIndexType());

  // get memref dim
  mlir::Value memrefDim;
  if (baseMemSTy.getDimSize(0) < 0) {
    mlir::Value dimIdx0 = ConstantPool::getValue(0, builder.getIndexType());
    // llvm::outs() << dimIdx0 << "\n";
    memrefDim =
        builder.create<mlir::memref::DimOp>(loc, baseMem, dimIdx0).getResult();
  } else {
    memrefDim = ConstantPool::getValue(baseMemSTy.getDimSize(0),
                                       builder.getIndexType());
  }

  // fix index out-of-bound with remu
  mlir::Value afterRemuIdx =
      builder.create<mlir::index::RemUOp>(loc, insertIdx, memrefDim)
          .getResult();

  // check and expand memref size if needed
  mlir::Value remainIdxSpace =
      builder.create<mlir::index::SubOp>(loc, memrefDim, afterRemuIdx)
          .getResult();
  mlir::Value havingEnoughIdxSpace =
      builder
          .create<mlir::index::CmpOp>(loc, mlir::index::IndexCmpPredicate::UGE,
                                      remainIdxSpace, vectorDimVal)
          .getResult();
  // Write the dynamic check and fix code.
  auto sameBranch = [&](mlir::OpBuilder &b, mlir::Location loc) {
    mlir::Operation *newSrcOp = FixUBUtils::castToAllDyn(builder, loc, baseMem);
    b.create<mlir::scf::YieldOp>(loc, newSrcOp->getResult(0));
  };
  auto notSameBranch = [&](mlir::OpBuilder &b, mlir::Location loc) {
    // expand memref size
    mlir::Value needSpace =
        builder.create<mlir::index::SubOp>(loc, vectorDimVal, remainIdxSpace)
            .getResult();
    mlir::Value newMemrefDim =
        builder.create<mlir::index::AddOp>(loc, memrefDim, needSpace)
            .getResult();
    // generate new memref
    llvm::SmallVector<mlir::Value> dynDimAttribute =
        llvm::SmallVector<mlir::Value>({newMemrefDim});
    llvm::SmallVector<int64_t> fixTargetShape =
        llvm::SmallVector<int64_t>({mlir::ShapedType::kDynamic});
    mlir::MemRefType newMemrefSTy =
        mlir::MemRefType::get(fixTargetShape, baseMemSTy.getElementType());
    mlir::Value newMemref =
        builder
            .create<mlir::memref::AllocOp>(loc, newMemrefSTy, dynDimAttribute)
            .getResult();
    mlir::Value randnumVal = FixUBUtils::genRandomIntOrFloatValue(
        builder, loc, newMemrefSTy.getElementType());
    builder.create<mlir::linalg::FillOp>(loc, randnumVal, newMemref);
    b.create<mlir::scf::YieldOp>(loc, newMemref);
  };
  mlir::Operation *scfIfOp =
      builder
          .create<mlir::scf::IfOp>(loc, havingEnoughIdxSpace, sameBranch,
                                   notSameBranch)
          .getOperation();

  op->setOperand(0, scfIfOp->getResult(0));
  op->setOperand(1, afterRemuIdx);
  return;
}