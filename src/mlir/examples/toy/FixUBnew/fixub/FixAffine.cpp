#include <string>

#include "llvm/Support/raw_ostream.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Affine/Utils.h"
// #include "mlir/Dialect/Index/IR/IndexEnums.h.inc"
#include "mlir/IR/Operation.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Parser/Parser.h"

#include "fixub/ConstantPool.h"
#include "fixub/DelOp.h"
#include "fixub/FixAffine.h"
#include "fixub/FixUBUtils.h"

void FixAffine::fixaffine(mlir::Operation *op) {
  std::string opname = op->getName().getStringRef().str();
  if (opname == "affine.load" || opname == "affine.store") {
    FixAffine::fixStoreLoad(op);
  }
  if (opname == "affine.vector_load" || opname == "affine.vector_store") {
    FixAffine::fixVectorStoreLoad(op);
  }
  if (opname == "affine.yield") {
    FixAffine::fixYield(op);
  }
  return;
}

// // this try to replace the invalid index by generating ramdom value right
// after const or memref defining value
// // failed, due to the wrong position of getting the defining op of the
// dynamic memref value mlir::Value FixAffine::genValidIndex(int64_t dimIdx,
// mlir::Value dimVal) {

//   mlir::Operation *defingOp = dimVal.getDefiningOp();

//   if(defingOp->getName().getStringRef().str() != "memref.dim") {

//     mlir::Location defingOpLoc = defingOp->getLoc();
//     mlir::OpBuilder defingOpBuilder(defingOp->getBlock(),
//     defingOp->getBlock()->begin());
//     defingOpBuilder.setInsertionPointAfter(defingOp);
//     mlir::Value randVal = defingOpBuilder
//                           .create<mlir::index::ConstantOp>(defingOpLoc,
//                           mlir::IntegerAttr::get(defingOpBuilder.getIndexType(),
//                           FixUBUtils::randsval(0, 100))) .getOperation()
//                           ->getResult(0);
//     return defingOpBuilder.create<mlir::index::RemUOp>(defingOpLoc, randVal,
//     dimVal).getResult();
//   }

//   mlir::Value memrefVal = defingOp->getOperand(0);
//   llvm::outs() << memrefVal << "\n";
//   if(memrefVal.getDefiningOp() == NULL) {
//     mlir::Location defingOpLoc = defingOp->getLoc();
//     mlir::OpBuilder defingOpBuilder(defingOp->getBlock(),
//     defingOp->getBlock()->begin()); return ConstantPool::getValue(0,
//     defingOpBuilder.getIndexType());
//   }

//   defingOp = memrefVal.getDefiningOp(); // bug here, can't find the right
//   place of the defining op, may because of delating op and fixub in
//   memref.alloc?
//   // if(defingOp->getName().getStringRef().str() == "memref.alloc" ||
//   defingOp->getName().getStringRef().str() == "memref.alloca") {

//   //   // llvm::iplist<Operation> blockOperations = ;
//   //   bool flag =false;
//   //   for(llvm::iplist<mlir::Operation>::iterator it =
//   defingOp->getBlock()->getOperations().begin(); it !=
//   defingOp->getBlock()->getOperations().end(); it++) {
//   //     if(it == defingOp->getIterator()) {
//   //       flag = true;
//   //       continue;
//   //     }
//   //     if(flag) {
//   //       defingOp = &(*it);
//   //       break;
//   //     }
//   //   }
//   // }
//   llvm::outs() << defingOp->getName().getStringRef().str()<< "\n";
//   mlir::Location defingOpLoc = defingOp->getLoc();
//   mlir::OpBuilder defingOpBuilder(defingOp->getBlock(),
//   defingOp->getBlock()->begin());
//   defingOpBuilder.setInsertionPointAfter(defingOp);
//   mlir::Value dimIdxVal = ConstantPool::getValue(dimIdx,
//   defingOpBuilder.getIndexType()); mlir::Value newDimVal =
//   defingOpBuilder.create<mlir::memref::DimOp>(defingOpLoc,
//   defingOp->getOperand(0), dimIdxVal).getResult(); mlir::Value randVal =
//   defingOpBuilder
//                         .create<mlir::index::ConstantOp>(defingOpLoc,
//                         mlir::IntegerAttr::get(defingOpBuilder.getIndexType(),
//                         FixUBUtils::randsval(0, 100))) .getOperation()
//                         ->getResult(0);
//   // delOps.push_back(dimVal.getDefiningOp());
//   return defingOpBuilder.create<mlir::index::RemUOp>(defingOpLoc, randVal,
//   newDimVal).getResult();

// }

bool FixAffine::isValidAffineIndexOperand(mlir::Value value,
                                          mlir::Region *region) {
  return mlir::affine::isValidDim(value, region) ||
         mlir::affine::isValidSymbol(value, region);
}
// to do: check the load/store attribute with sennbai
// now the above attribute has to be produced by index.constant,
// or will cause error "Affine.load op index must be a dimension or symbol
// identifier" when using index.remu to cal the legal attribute reference:
// https://discourse.llvm.org/t/affine-load-op-index-must-be-a-dimension-or-symbol-identifier/63647
void FixAffine::fixStoreLoad(mlir::Operation *op) {
  /**
   * Fix undefined behaviors in affine.load, affine.store.
   *
   * Example:
   *  affine.store %0, %alloc[%c25, %c25] : memref<13x?xi16>
   *  %1 = affine.load %alloc[%c25, %c25] : memref<13x?xi16>
   *
   * This fix work does the following tasks:
   * Fix the out of bound load/store behaviors in the two operations.
   *
   *
   * This fix replace the result of index remu dim size, as the new load/store
   * index to avoid the out of bound load/store behaviors.
   */
  // Set environment
  mlir::Location loc = op->getLoc();
  mlir::OpBuilder builder(op->getBlock(), op->getBlock()->begin());
  builder.setInsertionPoint(op);

  // Get the Operand
  mlir::Value dest;
  std::string opname = op->getName().getStringRef().str();
  if (opname == "affine.store")
    dest = op->getOperand(1);
  else
    dest = op->getOperand(0);
  // llvm::outs() << opname << "\n";
  // llvm::outs() << dest.getType() << "\n";

  mlir::ShapedType destSTy = dest.getType().dyn_cast<mlir::ShapedType>();

  // Get the required shape.
  llvm::SmallVector<mlir::Value> destShape = llvm::SmallVector<mlir::Value>();
  FixUBUtils::getShapes(builder, loc, dest, &destShape);

  // Get the load/store indices values
  llvm::SmallVector<mlir::Value> indicesAttrs;
  if (opname == "affine.store") {
    mlir::AffineMap affineMap =
        mlir::cast<mlir::affine::AffineStoreOp>(*op).getAffineMap();
    llvm::SmallVector<mlir::Value> mapOperands =
        mlir::cast<mlir::affine::AffineStoreOp>(*op).getMapOperands();
    indicesAttrs =
        *mlir::affine::expandAffineMap(builder, loc, affineMap, mapOperands);
  } else {
    mlir::AffineMap affineMap =
        mlir::cast<mlir::affine::AffineLoadOp>(*op).getAffineMap();
    llvm::SmallVector<mlir::Value> mapOperands =
        mlir::cast<mlir::affine::AffineLoadOp>(*op).getMapOperands();
    indicesAttrs =
        *mlir::affine::expandAffineMap(builder, loc, affineMap, mapOperands);
  }

  // mlir::AffineExpr affineMapExp = builder.getAffineDimExpr(0) %
  // builder.getAffineDimExpr(1); mlir::AffineMap affineMap =
  // mlir::AffineMap::get(2, 1, affineMapExp); mlir::AffineExpr affineMapExp =
  // builder.getAffineDimExpr(0) ; mlir::AffineMap affineMap =
  // mlir::AffineMap::get(1, 0, affineMapExp);

  // llvm::outs() << op->getNumOperands() << "\n";

  // For every index, cal the result of index remu dim size, as the new
  // load/store index
  llvm::SmallVector<mlir::Value> fixedIndicesAttrs =
      llvm::SmallVector<mlir::Value>();
  for (int64_t i = 0; i < indicesAttrs.size(); i++) {
    // mlir::Value dimIdxi = ConstantPool::getValue(i, builder.getIndexType());

    // mlir::Value memrefDim = builder.create<mlir::memref::DimOp>(loc, dest,
    // dimIdxi).getResult(); llvm::outs() <<
    // isValidAffineIndexOperand(memrefDim, scope) << " " <<
    // isValidAffineIndexOperand(indicesAttrs[i], scope) << "\n";
    mlir::Region *scope = mlir::affine::getAffineScope(op);
    mlir::Value remuVal =
        builder.create<mlir::index::RemUOp>(loc, indicesAttrs[i], destShape[i])
            .getResult();
    // llvm::outs() << isValidAffineIndexOperand(remuVal, scope) << "\n";

    // check if the fixed index is valid, if not, use constant 0 as fixed index
    // instead
    if (!isValidAffineIndexOperand(remuVal, scope)) {
      delOps.push_back(remuVal.getDefiningOp());
      fixedIndicesAttrs.push_back(
          ConstantPool::getValue(0, builder.getIndexType()));
    } else {
      fixedIndicesAttrs.push_back(remuVal);
    }
    // llvm::outs() <<
    // llvm::dyn_cast<mlir::BlockArgument>(remuVal).getOwner()->getParentOp()->hasTrait<mlir::OpTrait::AffineScope>()
    // << "\n"; llvm::SmallVector<mlir::Value> affineApplyOperands;
    // affineApplyOperands.push_back(remuVal);
    // affineApplyOperands.push_back(destShape[i]);
    // mlir::Value remuVal1 = builder.create<mlir::affine::AffineApplyOp>(loc,
    // affineMap, affineApplyOperands).getResult();

    // fixedIndicesAttrs.push_back(num0);
  }
  // llvm::outs() << "\n";
  // llvm::outs() << fixedIndicesAttrs.size() << "\n";

  // Use the new indices to construct new operation, and replace old operation
  mlir::Operation *newOp;
  if (opname == "affine.store")
    newOp = builder.create<mlir::affine::AffineStoreOp>(
        loc, op->getOperand(0), op->getOperand(1),
        mlir::ValueRange(fixedIndicesAttrs));
  else
    newOp = builder.create<mlir::affine::AffineLoadOp>(
        loc, op->getOperand(0), mlir::ValueRange(fixedIndicesAttrs));
  op->replaceAllUsesWith(newOp);
  delOps.push_back(op);
  // llvm::outs() << "\n" << resultOperands << "\n";

  return;
}

void FixAffine::fixVectorStoreLoad(mlir::Operation *op) {
  /**
   * Fix undefined behaviors in affine.load, affine.store.
   *
   * Example:
   *  affine.vector_store %64, %alloc[%c25, %c25] : memref<13x?xi16>,
   * vector<21xi16> %65 = affine.vector_load %alloc[%c25, %c25] :
   * memref<13x?xi16>, vector<21xi16>
   *
   * This fix work does the following tasks:
   * (1) Fix the out of bound load/store behaviors in the two operations.
   * (2) Fix the vector extract space out-of-bound error
   *
   *
   * This fix funtion writet set the index as ori index mod vector dim size
   * and generate new memref with enough space(expand the last dim) to extract
   */
  // Set environment
  mlir::Location loc = op->getLoc();
  mlir::OpBuilder builder(op->getBlock(), op->getBlock()->begin());
  builder.setInsertionPoint(op);

  // Get the Operand
  mlir::Value oriMemrefVal;
  mlir::Value vectorDimVal;
  std::string opname = op->getName().getStringRef().str();

  llvm::SmallVector<mlir::Value> indicesAttrs;
  if (opname == "affine.vector_store") {

    oriMemrefVal = op->getOperand(1);

    mlir::AffineMap affineMap =
        mlir::cast<mlir::affine::AffineVectorStoreOp>(*op).getAffineMap();
    llvm::SmallVector<mlir::Value> mapOperands =
        mlir::cast<mlir::affine::AffineVectorStoreOp>(*op).getMapOperands();
    indicesAttrs =
        *mlir::affine::expandAffineMap(builder, loc, affineMap, mapOperands);

    mlir::ShapedType vectorSTy =
        op->getOperand(0).getType().dyn_cast<mlir::ShapedType>();
    vectorDimVal =
        ConstantPool::getValue(vectorSTy.getDimSize(0), builder.getIndexType());
  } else {

    oriMemrefVal = op->getOperand(0);

    mlir::AffineMap affineMap =
        mlir::cast<mlir::affine::AffineVectorLoadOp>(*op).getAffineMap();
    llvm::SmallVector<mlir::Value> mapOperands =
        mlir::cast<mlir::affine::AffineVectorLoadOp>(*op).getMapOperands();
    indicesAttrs =
        *mlir::affine::expandAffineMap(builder, loc, affineMap, mapOperands);

    mlir::ShapedType vectorSTy =
        op->getResult(0).getType().dyn_cast<mlir::ShapedType>();
    vectorDimVal =
        ConstantPool::getValue(vectorSTy.getDimSize(0), builder.getIndexType());
  }

  mlir::ShapedType oriMemrefSTy =
      oriMemrefVal.getType().dyn_cast<mlir::ShapedType>();
  // Get the required shape.
  llvm::SmallVector<mlir::Value> oriMemrefShape =
      llvm::SmallVector<mlir::Value>();
  FixUBUtils::getShapes(builder, loc, oriMemrefVal, &oriMemrefShape);

  // fix index out-of-bound with remu
  llvm::SmallVector<mlir::Value> fixedIndicesAttrs =
      llvm::SmallVector<mlir::Value>();
  llvm::SmallVector<int64_t> fixTargetShape;
  for (int64_t i = 0; i < indicesAttrs.size(); i++) {

    mlir::Region *scope = mlir::affine::getAffineScope(op);
    mlir::Value remuVal = builder
                              .create<mlir::index::RemUOp>(loc, indicesAttrs[i],
                                                           oriMemrefShape[i])
                              .getResult();

    // check if the fixed index is valid, if not, use constant 0 as fixed index
    // instead
    if (!isValidAffineIndexOperand(remuVal, scope)) {
      delOps.push_back(remuVal.getDefiningOp());
      fixedIndicesAttrs.push_back(
          ConstantPool::getValue(0, builder.getIndexType()));
    } else {
      fixedIndicesAttrs.push_back(remuVal);
    }
    fixTargetShape.push_back(mlir::ShapedType::kDynamic);
  }

  // check and expand memref size if needed

  // cal the memref size and size before start index
  mlir::Value memrefSizeVal = ConstantPool::getValue(1, builder.getIndexType());
  mlir::Value afterRemuSizeVal =
      ConstantPool::getValue(0, builder.getIndexType());
  for (int64_t i = fixedIndicesAttrs.size() - 1; i >= 0; i--) {
    mlir::Value dimiUsed = builder
                               .create<mlir::index::MulOp>(loc, memrefSizeVal,
                                                           fixedIndicesAttrs[i])
                               .getResult();
    afterRemuSizeVal =
        builder.create<mlir::index::AddOp>(loc, afterRemuSizeVal, dimiUsed)
            .getResult();
    memrefSizeVal =
        builder
            .create<mlir::index::MulOp>(loc, memrefSizeVal, oriMemrefShape[i])
            .getResult();
  }

  // compare to check if the remaining space is enough
  mlir::Value remainIdxSpace =
      builder.create<mlir::index::SubOp>(loc, memrefSizeVal, afterRemuSizeVal)
          .getResult();
  mlir::Value havingEnoughIdxSpace =
      builder
          .create<mlir::index::CmpOp>(loc, mlir::index::IndexCmpPredicate::UGE,
                                      remainIdxSpace, vectorDimVal)
          .getResult();
  // Write the dynamic check and fix code.
  auto sameBranch = [&](mlir::OpBuilder &b, mlir::Location loc) {
    mlir::Operation *newSrcOp =
        FixUBUtils::castToAllDyn(builder, loc, oriMemrefVal);
    b.create<mlir::scf::YieldOp>(loc, newSrcOp->getResult(0));
  };
  auto notSameBranch = [&](mlir::OpBuilder &b, mlir::Location loc) {
    // expand memref size
    mlir::Value needSpace =
        builder.create<mlir::index::SubOp>(loc, vectorDimVal, remainIdxSpace)
            .getResult();
    oriMemrefShape[oriMemrefShape.size() - 1] =
        builder
            .create<mlir::index::AddOp>(
                loc, oriMemrefShape[oriMemrefShape.size() - 1], needSpace)
            .getResult();

    // generate new memref
    mlir::MemRefType newMemrefSTy =
        mlir::MemRefType::get(fixTargetShape, oriMemrefSTy.getElementType());
    mlir::Value newMemref =
        builder.create<mlir::memref::AllocOp>(loc, newMemrefSTy, oriMemrefShape)
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

  // Use the new indices to construct new operation, and replace old operation
  mlir::Operation *newOp;
  if (opname == "affine.vector_store")
    newOp = builder.create<mlir::affine::AffineVectorStoreOp>(
        loc, op->getOperand(0), scfIfOp->getResult(0),
        mlir::ValueRange(fixedIndicesAttrs));
  else {
    mlir::VectorType vectorTy =
        op->getResult(0).getType().dyn_cast<mlir::VectorType>();
    newOp = builder.create<mlir::affine::AffineVectorLoadOp>(
        loc, vectorTy, scfIfOp->getResult(0),
        mlir::ValueRange(fixedIndicesAttrs));
  }
  op->replaceAllUsesWith(newOp);
  delOps.push_back(op);
  return;
}

void FixAffine::fixYield(mlir::Operation *op) {
  /**
   * Fix undefined behaviors in affine.yield.
   * When the operands have dynamic shpae, such as tensor<?xi64> and
   * memref<?xi64>, the affine.yield operation may return tensor or memref value
   * that have different real dimension with the argument of affine.for. This
   * undefined behavior exists in MLIRSmith-generated MLIR programs.
   *
   * Example:
   * ...
   * %cast_28 = tensor.cast %from_elements_27 : tensor<2x1x1xi16> to
   * tensor<?x1x1xi16> %229 = affine.for %arg0 = 0 to 92 iter_args(%arg1 =
   * %cast_28) -> (tensor<?x1x1xi16>) {
   *  ...
   *  %cast_178 = tensor.cast %from_elements_177 : tensor<3x1x1xi16> to
   * tensor<?x1x1xi16> affine.yield %cast_178 : tensor<?x1x1xi16>
   * }
   * In the above MLIR code, %cast_178(tensor<3x1x1xi16>) has a mismatched type
   * with %arg1(tensor<2x1x1xi16>)
   *
   * This fix work does the following tasks:
   * (1) extract all of the dynamic shaped tensor or memref operands of
   * affine.yield. (2) for each dynamic shaped tensor or memref, extract their
   * dynamic dimensions. (3) compare the dynamic dimensions, and use scf.if to
   * generate all of the new operands.
   */
  // Set environment
  mlir::Location loc = op->getLoc();
  mlir::OpBuilder builder(op->getBlock(), op->getBlock()->begin());
  builder.setInsertionPoint(op);

  llvm::outs() << "affine.yield fix start \n";

  // 1. extract all of the dynamic shaped tensor of affine.yield operands
  llvm::SmallVector<int64_t> targetOperandIndices =
      llvm::SmallVector<int64_t>();
  llvm::SmallVector<mlir::Value> yieldOperands =
      llvm::SmallVector<mlir::Value>();
  for (int64_t i = 0; i < op->getNumOperands(); ++i) {
    mlir::Value operand = op->getOperand(i);
    mlir::Type operandTy = operand.getType();

    // llvm::outs() << mlir::isa<mlir::ShapedType>(operandTy) << "\n";
    // llvm::outs() << FixUBUtils::isDynShape(
    //                     mlir::dyn_cast<mlir::ShapedType>(operandTy))
    //              << "\n";

    if (mlir::isa<mlir::ShapedType>(operandTy) &&
        FixUBUtils::isDynShape(mlir::dyn_cast<mlir::ShapedType>(operandTy))) {
      yieldOperands.push_back(operand);
      targetOperandIndices.push_back(i);
    }
  }

  // if there is no such dynamic shaped tensor, just return.
  if (!yieldOperands.size())
    return;

  mlir::Operation *affineOp = op->getParentOp();
  while (!mlir::isa<mlir::affine::AffineForOp>(affineOp) &&
         !mlir::isa<mlir::affine::AffineIfOp>(affineOp) &&
         !mlir::isa<mlir::affine::AffineParallelOp>(affineOp))
    affineOp = affineOp->getParentOp();

  llvm::SmallVector<mlir::Value> affineOperands =
      llvm::SmallVector<mlir::Value>();
  for (int64_t i = 0; i < targetOperandIndices.size(); ++i)
    affineOperands.push_back(affineOp->getOperand(i));

  // 2. for each dynamic tensor/memref of affine.yield operands.
  for (int64_t i = 0; i < targetOperandIndices.size(); ++i) {
    // 2.1 extract all of the dynamic dimensions
    llvm::SmallVector<mlir::Value> yieldOperandDims =
        llvm::SmallVector<mlir::Value>();
    llvm::SmallVector<mlir::Value> affineOperandDims =
        llvm::SmallVector<mlir::Value>();
    FixUBUtils::getDynDims(builder, loc, yieldOperands[i], &yieldOperandDims);
    FixUBUtils::getDynDims(builder, loc, affineOperands[i], &affineOperandDims);
    // 2.2 compare the dims
    mlir::Operation *sameDim = FixUBUtils::compareShape(
        builder, loc, yieldOperandDims, affineOperandDims);
    // 2.3 return the argument itself if the dimension of the affine.yield
    // mismatched with dimension of affine.for/if/parapell...

    auto sameBuilder = [&](mlir::OpBuilder &b, mlir::Location loc) {
      b.create<mlir::scf::YieldOp>(loc, yieldOperands[i]);
    };
    auto diffBuilder = [&](mlir::OpBuilder &b, mlir::Location loc) {
      b.create<mlir::scf::YieldOp>(loc, affineOperands[i]);
    };
    mlir::Operation *ifOp =
        builder
            .create<mlir::scf::IfOp>(loc, sameDim->getResult(0), sameBuilder,
                                     diffBuilder)
            .getOperation();
    op->setOperand(targetOperandIndices[i], ifOp->getResult(0));
  }

  llvm::outs() << "affine.yield fix start \n";

  return;
}