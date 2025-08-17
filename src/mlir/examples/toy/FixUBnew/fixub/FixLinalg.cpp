#include <string>

#include "llvm/Support/raw_ostream.h"

#include "mlir/Dialect/Affine/Passes.h"
// #include "mlir/Dialect/Index/IR/IndexEnums.h.inc"
#include "mlir/IR/Operation.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Parser/Parser.h"

#include "fixub/ConstantPool.h"
#include "fixub/FixLinalg.h"
#include "fixub/FixUBUtils.h"

void FixLinalg::fixlinalg(mlir::Operation *op) {
  std::string opname = op->getName().getStringRef().str();
  if (opname == "linalg.copy") {
    FixLinalg::fixCopy(op);
  }
  if (opname == "linalg.broadcast") {
    FixLinalg::fixBroadcast(op);
  }
  if (opname == "linalg.generic") {
    FixLinalg::fixGeneric(op);
  }
  if (opname == "linalg.map") {
    FixLinalg::fixMap(op);
  }
  if (opname == "linalg.matmul") {
    FixLinalg::fixMatmul(op);
  }
  if (opname == "linalg.transpose") {
    FixLinalg::fixTranspose(op);
  }
  return;
}

void FixLinalg::fixCopy(mlir::Operation *op) {
  /**
   * Fix undefined behaviors in linalg.copy.
   *
   * Example:
   *  linalg.copy ins(%0 : tensor<?x4x?x6xi32>) outs(%1 : tensor<4x?x6x?xi32>)
   * -> tensor<4x?x6x?xi32>
   *
   * The followinng inputs will be rejected by static semantic checking of
   * mlir-opt: (1) tensors(memrefs) that have differnet ranks (2)
   * tensors(memrefs) that have at least one pair of static dimensions that are
   * not same.
   *
   *
   * Note that: the element type is checked during static semantic checking.
   *
   * This fix funtion writet the dynamic check and fix code to ensuring the
   * shape of src tensor(memref) and dest tensor(memref).
   */
  // Set environment
  mlir::Location loc = op->getLoc();
  mlir::OpBuilder builder(op->getBlock(), op->getBlock()->begin());
  builder.setInsertionPoint(op);

  // Get the Operand
  mlir::Value src = op->getOperand(0);
  mlir::Value dest = op->getOperand(1);
  mlir::ShapedType srcSTy = src.getType().dyn_cast<mlir::ShapedType>();
  mlir::ShapedType destSTy = dest.getType().dyn_cast<mlir::ShapedType>();

  // Get the required shape.
  llvm::SmallVector<mlir::Value> srcShape = llvm::SmallVector<mlir::Value>();
  llvm::SmallVector<mlir::Value> destShape = llvm::SmallVector<mlir::Value>();
  FixUBUtils::getShapes(builder, loc, src, &srcShape);
  FixUBUtils::getShapes(builder, loc, dest, &destShape);

  // Compare the shape.
  mlir::Operation *sameOp =
      FixUBUtils::compareShape(builder, loc, srcShape, destShape);
  mlir::Value sameShape = sameOp->getResult(0);

  // Write the dynamic check and fix code.
  auto sameBranch = [&](mlir::OpBuilder &b, mlir::Location loc) {
    mlir::Operation *newSrcOp = FixUBUtils::castToAllDyn(builder, loc, src);
    b.create<mlir::scf::YieldOp>(loc, newSrcOp->getResult(0));
  };
  auto notSameBranch = [&](mlir::OpBuilder &b, mlir::Location loc) {
    llvm::SmallVector<mlir::Value> dynDims = llvm::SmallVector<mlir::Value>();
    llvm::SmallVector<int64_t> dynDimsIdices =
        llvm::SmallVector<int64_t>(); // this is unused here because the src and
                                      // dest shape should be strictly same
    FixUBUtils::getDynDims(builder, loc, dest, &dynDims, &dynDimsIdices);
    mlir::Value newSrc;
    // llvm::outs() << "fix linalg.copy"
    if (llvm::isa<mlir::MemRefType>(src.getType())) {
      newSrc = builder.create<mlir::memref::AllocOp>(
          loc, destSTy.dyn_cast<mlir::MemRefType>(), dynDims);
      mlir::Value randnumVal = FixUBUtils::genRandomIntOrFloatValue(
          builder, loc, destSTy.dyn_cast<mlir::MemRefType>().getElementType());
      builder.create<mlir::linalg::FillOp>(loc, randnumVal, newSrc);
    } else {
      mlir::Value randnumVal = FixUBUtils::genRandomIntOrFloatValue(
          builder, loc, destSTy.getElementType());
      newSrc = builder.create<mlir::tensor::SplatOp>(loc, randnumVal, destSTy,
                                                     dynDims);
    }

    mlir::Operation *newSrcOp = FixUBUtils::castToAllDyn(builder, loc, newSrc);
    b.create<mlir::scf::YieldOp>(loc, newSrcOp->getResult(0));
  };
  mlir::Operation *scfIfOp =
      builder.create<mlir::scf::IfOp>(loc, sameShape, sameBranch, notSameBranch)
          .getOperation();
  op->setOperand(0, scfIfOp->getResult(0));
  return;
}

void FixLinalg::fixBroadcast(mlir::Operation *op) {
  /**
   * Fix undefined behaviors in linalg.broadcast.
   *
   * Example:
   *  linalg.broadcast ins(%4 : tensor<?x1xi64>) outs(%cast_2 :
   * tensor<?x1x?xi64>) dimensions = [2]
   *
   * The followinng inputs will be rejected by static semantic checking of
   * mlir-opt: (1) tensors(memrefs) that have differnet ranks (2)
   * tensors(memrefs) that have at least one pair of static dimensions that are
   * not same (3) tensors(memrefs) that have at least one pair of
   * dynamic-static(static-dynamic) pair
   *
   *
   * Note that: the element type is checked during static semantic checking.
   *
   * This fix funtion writet the dynamic check and fix code to ensuring the
   * shape of src tensor(memref) and dest tensor(memref) (without the splatting
   * dimensions).
   */
  // Set environment
  mlir::Location loc = op->getLoc();
  mlir::OpBuilder builder(op->getBlock(), op->getBlock()->begin());
  builder.setInsertionPoint(op);

  // Get the Operand
  mlir::Value src = op->getOperand(0);
  mlir::Value dest = op->getOperand(1);
  mlir::ShapedType srcSTy = src.getType().dyn_cast<mlir::ShapedType>();
  mlir::ShapedType destSTy = dest.getType().dyn_cast<mlir::ShapedType>();

  // Get the required shape.
  llvm::SmallVector<mlir::Value> srcShape = llvm::SmallVector<mlir::Value>();
  llvm::SmallVector<mlir::Value> destShapetmp =
      llvm::SmallVector<mlir::Value>(); // with the splatting dimensions
  FixUBUtils::getShapes(builder, loc, src, &srcShape);
  FixUBUtils::getShapes(builder, loc, dest, &destShapetmp);

  // Delete the splatting dimensions
  // to match the static and dynamic dimensions of the src type to the dest
  // type, but the two type are not exactly equal, needs to delete the splatting
  // dimensions
  llvm::ArrayRef<long int> addingDim =
      mlir::cast<mlir::linalg::BroadcastOp>(*op).getDimensions();
  llvm::SmallVector<mlir::Value> destShape =
      llvm::SmallVector<mlir::Value>();      // without the splatting dimensions
  llvm::SmallVector<int64_t> fixTargetShape; // without the splatting dimensions
  // llvm::outs() << "destShapetmp size : " << destShapetmp.size() << "\n";
  for (int64_t i = 0; i < destShapetmp.size(); i++) {
    // llvm::outs() << addingDim[i] << " ";
    if (!std::count(addingDim.begin(), addingDim.end(), i)) {
      destShape.push_back(destShapetmp[i]);
      if (destSTy.getDimSize(i) < 0)
        fixTargetShape.push_back(mlir::ShapedType::kDynamic);
      else
        fixTargetShape.push_back(destSTy.getDimSize(i));
      // llvm::outs() << "destShape push back : " << i << "\n";
    }
  }
  // for (int64_t i = 0; i < fixTargetShape.size(); i++) {
  //   llvm::outs() << fixTargetShape[i] << " ";
  //   // destShape.erase(addingDim[i]);
  // }
  // llvm::outs() << "\n";

  // Compare the shape.
  mlir::Operation *sameOp =
      FixUBUtils::compareShape(builder, loc, srcShape, destShape);
  mlir::Value sameShape = sameOp->getResult(0);

  // Write the dynamic check and fix code.
  auto sameBranch = [&](mlir::OpBuilder &b, mlir::Location loc) {
    // this operation restricts the input and output shape to have same static
    // dimensions, or both dynamic dimensions, so castToAllDynis unnecessary
    // mlir::Operation *newSrcOp =
    //     FixUBUtils::castToAllDyn(builder, loc, src);
    b.create<mlir::scf::YieldOp>(loc, src); // newSrcOp->getResult(0));
  };
  auto notSameBranch = [&](mlir::OpBuilder &b, mlir::Location loc) {
    // geting dynamic dims attributes
    llvm::SmallVector<mlir::Value> destDynDims =
        llvm::SmallVector<mlir::Value>();
    llvm::SmallVector<int64_t> destDynDimsIdices = llvm::SmallVector<int64_t>();
    FixUBUtils::getDynDims(builder, loc, dest, &destDynDims,
                           &destDynDimsIdices);

    // for the not splatting dynamic dimensions, add this dim imformation in the
    // attribute of memref.alloc/tensor.splat
    llvm::SmallVector<mlir::Value> dynDims = llvm::SmallVector<mlir::Value>();
    for (int64_t i = 0; i < destDynDims.size(); i++) {
      if (!std::count(addingDim.begin(), addingDim.end(),
                      destDynDimsIdices[i])) {
        dynDims.push_back(destDynDims[i]);
        // llvm::outs() << "adding dynDim of dim " << destDynDimsIdices[i] <<
        // "\n";
      }
    }

    // generating ub free dest shape
    mlir::Value newSrc;
    // llvm::outs() << "fix linalg.copy"
    if (llvm::isa<mlir::MemRefType>(src.getType())) {
      mlir::MemRefType newDestSTy =
          mlir::MemRefType::get(fixTargetShape, destSTy.getElementType());
      newSrc = builder.create<mlir::memref::AllocOp>(loc, newDestSTy, dynDims);

      mlir::Value randnumVal = FixUBUtils::genRandomIntOrFloatValue(
          builder, loc, newDestSTy.getElementType());
      builder.create<mlir::linalg::FillOp>(loc, randnumVal, newSrc);
    } else {
      mlir::RankedTensorType newDestSTy =
          mlir::RankedTensorType::get(fixTargetShape, destSTy.getElementType());
      mlir::Value randnumVal = FixUBUtils::genRandomIntOrFloatValue(
          builder, loc, destSTy.getElementType());
      newSrc = builder.create<mlir::tensor::SplatOp>(loc, randnumVal,
                                                     newDestSTy, dynDims);
    }

    // this operation restricts the input and output shape to have same static
    // dimensions, or both dynamic dimensions, so castToAllDynis unnecessary
    // mlir::Operation *newSrcOp =
    //     FixUBUtils::castToAllDyn(builder, loc, newSrc);
    b.create<mlir::scf::YieldOp>(loc, newSrc); // newSrcOp->getResult(0));
  };
  mlir::Operation *scfIfOp =
      builder.create<mlir::scf::IfOp>(loc, sameShape, sameBranch, notSameBranch)
          .getOperation();
  op->setOperand(0, scfIfOp->getResult(0));
  return;
}

void FixLinalg::fixGeneric(mlir::Operation *op) {
  /**
   * Fix undefined behaviors in linalg.map.
   *
   * Example:
   *  %14 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1,
   * d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) ->
   * (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%6,
   * %13 : tensor<?x?x?xi16>, tensor<?x?x?xi16>) outs(%cast_3 : tensor<?x5xi16>)
   * { ^bb0(%in: i16, %in_8: i16, %out: i16): linalg.yield %in : i16 } ->
   * tensor<?x5xi16>
   *
   * The followinng inputs will be rejected by static semantic checking of
   * mlir-opt: (1) tensors(memrefs) that have differnet ranks (2)
   * tensors(memrefs) that have at least one pair of static dimensions that are
   * not same
   *
   *
   * Note that: the element type is checked during static semantic checking.
   *
   * This fix funtion writet the dynamic check and fix code to ensuring the
   * shape of src tensor(memref) and dest tensor(memref).
   */
  // Set environment
  mlir::Location loc = op->getLoc();
  mlir::OpBuilder builder(op->getBlock(), op->getBlock()->begin());
  builder.setInsertionPoint(op);

  // Get the Operand
  llvm::SmallVector<mlir::Value> inputs;
  std::string opname = op->getName().getStringRef().str();
  inputs = mlir::cast<mlir::linalg::GenericOp>(*op).getInputs();

  mlir::Value dest = op->getOperand(inputs.size());
  mlir::ShapedType destSTy = dest.getType().dyn_cast<mlir::ShapedType>();

  // Get the required shape.
  mlir::Value src1 = inputs[0];
  mlir::ShapedType src1STy = src1.getType().dyn_cast<mlir::ShapedType>();
  // for(int i=0 ;i< inputs.size(); i++) {
  //   llvm::outs() << inputs[i].getType()<<" ";
  // }
  // llvm::outs()<<"\n";

  llvm::SmallVector<mlir::Value> src1Shape = llvm::SmallVector<mlir::Value>();
  llvm::SmallVector<mlir::Value> destShape = llvm::SmallVector<mlir::Value>();
  FixUBUtils::getShapes(builder, loc, src1, &src1Shape);
  FixUBUtils::getShapes(builder, loc, dest, &destShape);

  // using the dimsize in dest for parallel and the dimsize in src1 for
  // reduction to form the fix target shape and the tocompare shape
  llvm::SmallVector<mlir::utils::IteratorType> itType =
      mlir::cast<mlir::linalg::GenericOp>(*op).getIteratorTypesArray();
  llvm::SmallVector<mlir::Value> toCompareShape =
      llvm::SmallVector<mlir::Value>();
  llvm::SmallVector<int64_t> fixTargetShape;
  int64_t src1i = 0, desti = 0;
  while (src1i < src1Shape.size()) {

    if (itType[src1i] == mlir::utils::IteratorType::parallel) {
      toCompareShape.push_back(destShape[desti]);

      if (destSTy.getDimSize(desti) < 0)
        fixTargetShape.push_back(mlir::ShapedType::kDynamic);
      else
        fixTargetShape.push_back(destSTy.getDimSize(desti));
      src1i++;
      desti++;
    } else {
      toCompareShape.push_back(src1Shape[src1i]);

      if (src1STy.getDimSize(src1i) < 0)
        fixTargetShape.push_back(mlir::ShapedType::kDynamic);
      else
        fixTargetShape.push_back(src1STy.getDimSize(src1i));
      src1i++;
    }
  }
  // for(int i=0 ;i< src1Shape.size(); i++) {
  //   llvm::outs() << fixTargetShape[i] << " " << toCompareShape[i] << "\n";
  // }
  // llvm::outs()<<"\n";

  // // fix every input shape
  for (int64_t i = 0; i < inputs.size(); i++) {

    mlir::Value src = inputs[i];
    // llvm::outs() << src.getType() << "\n";
    mlir::ShapedType srcSTy = src.getType().dyn_cast<mlir::ShapedType>();
    llvm::SmallVector<mlir::Value> srcShape = llvm::SmallVector<mlir::Value>();
    FixUBUtils::getShapes(builder, loc, src, &srcShape);
    // for(int i = 0; i < srcShape.size(); i++) {
    //   llvm::outs() << srcShape[i]  << "\n";
    // }
    // llvm::outs()<<"\n";
    // for(int i = 0; i < fixTargetShape.size(); i++) {
    //   llvm::outs() << fixTargetShape[i] << "\n";
    // }
    // llvm::outs()<<"\n";
    // Compare the shape.
    mlir::Operation *sameOp =
        FixUBUtils::compareShape(builder, loc, srcShape, toCompareShape);
    mlir::Value sameShape = sameOp->getResult(0);

    // Write the dynamic check and fix code.
    auto sameBranch = [&](mlir::OpBuilder &b, mlir::Location loc) {
      mlir::Operation *newSrcOp = FixUBUtils::castToAllDyn(builder, loc, src);
      b.create<mlir::scf::YieldOp>(loc, newSrcOp->getResult(0));
    };
    auto notSameBranch = [&](mlir::OpBuilder &b, mlir::Location loc) {
      llvm::SmallVector<mlir::Value> src1DynDims =
          llvm::SmallVector<mlir::Value>();
      llvm::SmallVector<int64_t> src1DynDimsIdices =
          llvm::SmallVector<int64_t>();
      FixUBUtils::getDynDims(builder, loc, src1, &src1DynDims,
                             &src1DynDimsIdices);
      llvm::SmallVector<mlir::Value> destDynDims =
          llvm::SmallVector<mlir::Value>();
      llvm::SmallVector<int64_t> destDynDimsIdices =
          llvm::SmallVector<int64_t>();
      FixUBUtils::getDynDims(builder, loc, dest, &destDynDims,
                             &destDynDimsIdices);

      // using the dynamic dims in dest for parallel and the dynamic dims in
      // src1 for reduction for the dynamic dim attribute in the replacing value
      llvm::SmallVector<mlir::Value> dynDimsForGen =
          llvm::SmallVector<mlir::Value>();
      int64_t src1i = 0, desti = 0;
      while (src1i < src1Shape.size()) {

        // llvm::outs() << "src1i: " << src1i << " desti: " << desti << "\n";
        if (itType[src1i] == mlir::utils::IteratorType::parallel) {

          if (fixTargetShape[src1i] > 0) {
            src1i++;
            desti++;
            continue;
          }
          int64_t dynIdx =
              std::distance(destDynDimsIdices.begin(),
                            std::find(destDynDimsIdices.begin(),
                                      destDynDimsIdices.end(), desti));
          // llvm::outs() << "parallel : " << dynIdx << "\n";
          dynDimsForGen.push_back(destDynDims[dynIdx]);
          src1i++;
          desti++;
        } else {
          if (fixTargetShape[src1i] > 0) {
            src1i++;
            continue;
          }
          int64_t dynIdx =
              std::distance(src1DynDimsIdices.begin(),
                            std::find(src1DynDimsIdices.begin(),
                                      src1DynDimsIdices.end(), src1i));
          // llvm::outs() << "reduce : " << dynIdx << "\n";
          dynDimsForGen.push_back(src1DynDims[dynIdx]);
          src1i++;
        }
      }

      // generating the replacing value
      mlir::Value newSrc;
      // llvm::outs() << "fix linalg.copy"
      if (llvm::isa<mlir::MemRefType>(src.getType())) {
        mlir::MemRefType newDestSTy =
            mlir::MemRefType::get(fixTargetShape, destSTy.getElementType());
        newSrc = builder.create<mlir::memref::AllocOp>(loc, newDestSTy,
                                                       dynDimsForGen);

        mlir::Value randnumVal = FixUBUtils::genRandomIntOrFloatValue(
            builder, loc, newDestSTy.getElementType());
        builder.create<mlir::linalg::FillOp>(loc, randnumVal, newSrc);
      } else {
        mlir::RankedTensorType newDestSTy = mlir::RankedTensorType::get(
            fixTargetShape, destSTy.getElementType());
        mlir::Value randnumVal = FixUBUtils::genRandomIntOrFloatValue(
            builder, loc, destSTy.getElementType());
        newSrc = builder.create<mlir::tensor::SplatOp>(
            loc, randnumVal, newDestSTy, dynDimsForGen);
      }
      mlir::Operation *newSrcOp =
          FixUBUtils::castToAllDyn(builder, loc, newSrc);
      b.create<mlir::scf::YieldOp>(loc, newSrcOp->getResult(0));
    };
    mlir::Operation *scfIfOp =
        builder
            .create<mlir::scf::IfOp>(loc, sameShape, sameBranch, notSameBranch)
            .getOperation();
    op->setOperand(i, scfIfOp->getResult(0));
  }

  return;
}

void FixLinalg::fixMap(mlir::Operation *op) {
  /**
   * Fix undefined behaviors in linalg.map.
   *
   * Example:
   *  %mapped = linalg.map ins(%3, %3, %3 : tensor<18x?x1xi16>,
   * tensor<18x?x1xi16>, tensor<18x?x1xi16>) outs(%49 : tensor<18x?x1xi16>)
   *  (%in: i16, %in_31: i16, %in_32: i16) {
   *
   *    %111 = arith.constant 0 : i16
   *    linalg.yield %111 : i16
   *    }
   *
   * The followinng inputs will be rejected by static semantic checking of
   * mlir-opt: (1) tensors(memrefs) that have differnet ranks (2)
   * tensors(memrefs) that have at least one pair of static dimensions that are
   * not same (3) tensors(memrefs) that have at least one pair of
   * dynamic-static(static-dynamic) pair.
   *
   *
   * Note that: the element type is checked during static semantic checking.
   *
   * This fix funtion writet the dynamic check and fix code to ensuring the
   * shape of src tensor(memref) and dest tensor(memref).
   */
  // Set environment
  mlir::Location loc = op->getLoc();
  mlir::OpBuilder builder(op->getBlock(), op->getBlock()->begin());
  builder.setInsertionPoint(op);

  // Get the Operand
  llvm::SmallVector<mlir::Value> inputs;
  std::string opname = op->getName().getStringRef().str();
  inputs = mlir::cast<mlir::linalg::MapOp>(*op).getInputs();

  mlir::Value dest = op->getOperand(inputs.size());
  mlir::ShapedType destSTy = dest.getType().dyn_cast<mlir::ShapedType>();

  // fix every input shape
  for (int64_t i = 0; i < inputs.size(); i++) {

    mlir::Value src = inputs[i];
    mlir::ShapedType srcSTy = src.getType().dyn_cast<mlir::ShapedType>();

    // for(int i=0 ;i< inputs.size(); i++) {
    //   llvm::outs() << inputs[i].getType()<<" ";
    // }
    // llvm::outs()<<"\n";

    // Get the required shape.
    llvm::SmallVector<mlir::Value> srcShape = llvm::SmallVector<mlir::Value>();
    llvm::SmallVector<mlir::Value> destShape = llvm::SmallVector<mlir::Value>();
    FixUBUtils::getShapes(builder, loc, src, &srcShape);
    FixUBUtils::getShapes(builder, loc, dest, &destShape);

    // Compare the shape.
    mlir::Operation *sameOp =
        FixUBUtils::compareShape(builder, loc, srcShape, destShape);
    mlir::Value sameShape = sameOp->getResult(0);

    // Write the dynamic check and fix code.
    auto sameBranch = [&](mlir::OpBuilder &b, mlir::Location loc) {
      // mlir::Operation *newSrcOp =
      //     FixUBUtils::castToAllDyn(builder, loc, src);
      b.create<mlir::scf::YieldOp>(loc, src); // newSrcOp->getResult(0));
    };
    auto notSameBranch = [&](mlir::OpBuilder &b, mlir::Location loc) {
      llvm::SmallVector<mlir::Value> dynDims = llvm::SmallVector<mlir::Value>();
      llvm::SmallVector<int64_t> dynDimsIdices =
          llvm::SmallVector<int64_t>(); // this is unused here because the src
                                        // and dest shape should be strictly
                                        // same
      FixUBUtils::getDynDims(builder, loc, dest, &dynDims, &dynDimsIdices);
      mlir::Value newSrc;
      // llvm::outs() << "fix linalg.copy"
      if (llvm::isa<mlir::MemRefType>(src.getType())) {
        // mlir::MemRefType newDestSTy = mlir::MemRefType::get(destSTy,
        // destSTy.getElementType());
        newSrc = builder.create<mlir::memref::AllocOp>(
            loc, destSTy.dyn_cast<mlir::MemRefType>(), dynDims);

        mlir::Value randnumVal = FixUBUtils::genRandomIntOrFloatValue(
            builder, loc,
            destSTy.dyn_cast<mlir::MemRefType>().getElementType());
        builder.create<mlir::linalg::FillOp>(loc, randnumVal, newSrc);
      } else {
        mlir::Value randnumVal = FixUBUtils::genRandomIntOrFloatValue(
            builder, loc, destSTy.getElementType());
        newSrc = builder.create<mlir::tensor::SplatOp>(loc, randnumVal, destSTy,
                                                       dynDims);
      }

      // mlir::Operation *newSrcOp =
      //     FixUBUtils::castToAllDyn(builder, loc, newSrc);
      b.create<mlir::scf::YieldOp>(loc, newSrc); // newSrcOp->getResult(0));
    };
    mlir::Operation *scfIfOp =
        builder
            .create<mlir::scf::IfOp>(loc, sameShape, sameBranch, notSameBranch)
            .getOperation();
    op->setOperand(i, scfIfOp->getResult(0));
  }

  return;
}

void FixLinalg::fixMatmul(mlir::Operation *op) {
  /**
   * Fix undefined behaviors in linalg.matmul.
   *
   * Example:
   *  %0 = linalg.matmul ins(%cast, %splat_0 : tensor<?x9xi32>,
   * tensor<9x24xi32>) outs(%cast_2 : tensor<?x24xi32>) -> tensor<?x24xi32>
   *
   * The followinng inputs will be rejected by static semantic checking of
   * mlir-opt: (1) tensors(memrefs) that have differnet ranks (2)
   * tensors(memrefs) that have at least one pair of static dimensions that are
   * not same.
   *
   *
   * Note that: the element type is checked during static semantic checking.
   *
   * This fix funtion writet the dynamic check and fix code to ensuring the
   * shape of src tensor(memref) and dest tensor(memref).
   */
  // Set environment
  mlir::Location loc = op->getLoc();
  mlir::OpBuilder builder(op->getBlock(), op->getBlock()->begin());
  builder.setInsertionPoint(op);

  // Get the Operand
  mlir::Value src1 = op->getOperand(0);
  mlir::Value src2 = op->getOperand(1);
  mlir::Value dest = op->getOperand(2);
  mlir::ShapedType src1STy = src1.getType().dyn_cast<mlir::ShapedType>();
  mlir::ShapedType src2STy = src2.getType().dyn_cast<mlir::ShapedType>();
  mlir::ShapedType destSTy = dest.getType().dyn_cast<mlir::ShapedType>();

  // Get the required shape.
  llvm::SmallVector<mlir::Value> src1Shape = llvm::SmallVector<mlir::Value>();
  llvm::SmallVector<mlir::Value> src2Shape = llvm::SmallVector<mlir::Value>();
  llvm::SmallVector<mlir::Value> destShape = llvm::SmallVector<mlir::Value>();
  FixUBUtils::getShapes(builder, loc, src1, &src1Shape);
  FixUBUtils::getShapes(builder, loc, src2, &src2Shape);
  FixUBUtils::getShapes(builder, loc, dest, &destShape);

  // using the shape of dest and dimsize of src1 dimension1 to form the fix
  // target shape of src1 and src2 using in genernate new shape
  llvm::SmallVector<int64_t> src1FixTargetShape, src2FixTargetShape;
  if (destSTy.getDimSize(0) < 0) {
    src1FixTargetShape.push_back(mlir::ShapedType::kDynamic);
  } else {
    src1FixTargetShape.push_back(destSTy.getDimSize(0));
  }
  if (src1STy.getDimSize(1) < 0) {
    src1FixTargetShape.push_back(mlir::ShapedType::kDynamic);
    src2FixTargetShape.push_back(mlir::ShapedType::kDynamic);
  } else {
    src1FixTargetShape.push_back(src1STy.getDimSize(1));
    src2FixTargetShape.push_back(src1STy.getDimSize(1));
  }
  if (destSTy.getDimSize(1) < 0) {
    src2FixTargetShape.push_back(mlir::ShapedType::kDynamic);
  } else {
    src2FixTargetShape.push_back(destSTy.getDimSize(1));
  }

  // the operation has 3 restraint, as for the i, j, k dimension restraint in
  // a[i][k] * b[k][j] = c[i][j] using the c[i][j] and k in a[i][k] for fixing
  // standard

  // for src1, should match the i dimension restraint
  mlir::Value sameShape = ConstantPool::getValue(1, builder.getI1Type());
  mlir::Value CmpDimI =
      builder
          .create<mlir::index::CmpOp>(loc, mlir::index::IndexCmpPredicate::EQ,
                                      src1Shape[0], destShape[0])
          .getResult();
  sameShape =
      builder.create<mlir::arith::AndIOp>(loc, CmpDimI, sameShape).getResult();

  // Write the dynamic check and fix code.
  auto src1SameBranch = [&](mlir::OpBuilder &b, mlir::Location loc) {
    mlir::Operation *newSrcOp = FixUBUtils::castToAllDyn(builder, loc, src1);
    b.create<mlir::scf::YieldOp>(loc, newSrcOp->getResult(0));
  };
  auto src1NotSameBranch = [&](mlir::OpBuilder &b, mlir::Location loc) {
    llvm::SmallVector<mlir::Value> dynDims = llvm::SmallVector<mlir::Value>();
    llvm::SmallVector<int64_t> dynDimsIdices = llvm::SmallVector<int64_t>();
    llvm::SmallVector<mlir::Value> dynDimsForGen =
        llvm::SmallVector<mlir::Value>();

    // push the i dim if is dynamic
    FixUBUtils::getDynDims(builder, loc, dest, &dynDims, &dynDimsIdices);
    if (dynDims.size() > 0)
      if (dynDimsIdices[0] == 0)
        dynDimsForGen.push_back(dynDims[0]);

    // push the k dim if is dynamic
    dynDims.clear();
    dynDimsIdices.clear();
    FixUBUtils::getDynDims(builder, loc, src1, &dynDims, &dynDimsIdices);
    if (dynDims.size() > 0)
      if (dynDimsIdices[dynDims.size() - 1] == 1)
        dynDimsForGen.push_back(dynDims[dynDims.size() - 1]);

    mlir::Value newSrc;
    // llvm::outs() << "fix linalg.copy"
    if (llvm::isa<mlir::MemRefType>(src2.getType())) {
      mlir::MemRefType newDestSTy =
          mlir::MemRefType::get(src1FixTargetShape, destSTy.getElementType());
      newSrc =
          builder.create<mlir::memref::AllocOp>(loc, newDestSTy, dynDimsForGen);

      mlir::Value randnumVal = FixUBUtils::genRandomIntOrFloatValue(
          builder, loc, newDestSTy.getElementType());
      builder.create<mlir::linalg::FillOp>(loc, randnumVal, newSrc);
    } else {
      mlir::RankedTensorType newDestSTy = mlir::RankedTensorType::get(
          src1FixTargetShape, destSTy.getElementType());
      mlir::Value randnumVal = FixUBUtils::genRandomIntOrFloatValue(
          builder, loc, destSTy.getElementType());
      newSrc = builder.create<mlir::tensor::SplatOp>(loc, randnumVal,
                                                     newDestSTy, dynDimsForGen);
    }

    mlir::Operation *newSrcOp = FixUBUtils::castToAllDyn(builder, loc, newSrc);
    b.create<mlir::scf::YieldOp>(loc, newSrcOp->getResult(0));
  };
  mlir::Operation *Src1ScfIfOp =
      builder
          .create<mlir::scf::IfOp>(loc, sameShape, src1SameBranch,
                                   src1NotSameBranch)
          .getOperation();
  op->setOperand(0, Src1ScfIfOp->getResult(0));

  // for src2, should match the j and k dimension restraints
  sameShape = ConstantPool::getValue(1, builder.getI1Type());
  mlir::Value CmpDimK =
      builder
          .create<mlir::index::CmpOp>(loc, mlir::index::IndexCmpPredicate::EQ,
                                      src1Shape[1], src2Shape[0])
          .getResult();
  sameShape =
      builder.create<mlir::arith::AndIOp>(loc, CmpDimK, sameShape).getResult();
  mlir::Value CmpDimJ =
      builder
          .create<mlir::index::CmpOp>(loc, mlir::index::IndexCmpPredicate::EQ,
                                      src2Shape[1], destShape[1])
          .getResult();
  sameShape =
      builder.create<mlir::arith::AndIOp>(loc, CmpDimJ, sameShape).getResult();

  // Write the dynamic check and fix code.
  auto src2SameBranch = [&](mlir::OpBuilder &b, mlir::Location loc) {
    mlir::Operation *newSrcOp = FixUBUtils::castToAllDyn(builder, loc, src2);
    b.create<mlir::scf::YieldOp>(loc, newSrcOp->getResult(0));
  };
  auto src2NotSameBranch = [&](mlir::OpBuilder &b, mlir::Location loc) {
    llvm::SmallVector<mlir::Value> dynDims = llvm::SmallVector<mlir::Value>();
    llvm::SmallVector<int64_t> dynDimsIdices =
        llvm::SmallVector<int64_t>(); // this is unused here because the src and
                                      // dest shape should be strictly same
    llvm::SmallVector<mlir::Value> dynDimsForGen =
        llvm::SmallVector<mlir::Value>();

    // push the k dim if is dynamic
    FixUBUtils::getDynDims(builder, loc, src1, &dynDims, &dynDimsIdices);
    if (dynDims.size() > 0)
      if (dynDimsIdices[dynDims.size() - 1] == 1)
        dynDimsForGen.push_back(dynDims[dynDims.size() - 1]);

    // push the j dim if is dynamic
    dynDims.clear();
    dynDimsIdices.clear();
    FixUBUtils::getDynDims(builder, loc, dest, &dynDims, &dynDimsIdices);
    if (dynDims.size() > 0)
      if (dynDimsIdices[dynDims.size() - 1] == 1)
        dynDimsForGen.push_back(dynDims[dynDims.size() - 1]);

    mlir::Value newSrc;
    // llvm::outs() << "fix linalg.copy"
    if (llvm::isa<mlir::MemRefType>(src2.getType())) {

      mlir::MemRefType newDestSTy =
          mlir::MemRefType::get(src2FixTargetShape, destSTy.getElementType());
      newSrc =
          builder.create<mlir::memref::AllocOp>(loc, newDestSTy, dynDimsForGen);

      mlir::Value randnumVal = FixUBUtils::genRandomIntOrFloatValue(
          builder, loc, newDestSTy.getElementType());
      builder.create<mlir::linalg::FillOp>(loc, randnumVal, newSrc);
    } else {
      mlir::RankedTensorType newDestSTy = mlir::RankedTensorType::get(
          src2FixTargetShape, destSTy.getElementType());
      mlir::Value randnumVal = FixUBUtils::genRandomIntOrFloatValue(
          builder, loc, destSTy.getElementType());
      newSrc = builder.create<mlir::tensor::SplatOp>(loc, randnumVal,
                                                     newDestSTy, dynDimsForGen);
    }

    mlir::Operation *newSrcOp = FixUBUtils::castToAllDyn(builder, loc, newSrc);
    b.create<mlir::scf::YieldOp>(loc, newSrcOp->getResult(0));
  };
  mlir::Operation *Src2ScfIfOp =
      builder
          .create<mlir::scf::IfOp>(loc, sameShape, src2SameBranch,
                                   src2NotSameBranch)
          .getOperation();
  op->setOperand(1, Src2ScfIfOp->getResult(0));

  return;
}

void FixLinalg::fixTranspose(mlir::Operation *op) {
  /**
   * Fix undefined behaviors in linalg.transpose.
   *
   * Example:
   *  %transposed = linalg.transpose ins(%6 : tensor<?x?x1xi32>) outs(%cast_3 :
   * tensor<1x?x?xi32>) permutation = [2, 1, 0]
   *
   * The followinng inputs will be rejected by static semantic checking of
   * mlir-opt: (1) tensors(memrefs) that have differnet ranks (2)
   * tensors(memrefs) that have at least one pair of static dimensions(the pair
   * of corresponding dimensions) that are not same (3) tensors(memrefs) that
   * have at least one pair of dynamic-static(static-dynamic) pair(the pair of
   * corresponding dimensions)
   *
   *
   * Note that: the element type is checked during static semantic checking.
   *
   * This fix funtion writet the dynamic check and fix code to ensuring the
   * shape of src tensor(memref) and dest tensor(memref) .
   */
  // Set environment
  mlir::Location loc = op->getLoc();
  mlir::OpBuilder builder(op->getBlock(), op->getBlock()->begin());
  builder.setInsertionPoint(op);

  // Get the Operand
  mlir::Value src = op->getOperand(0);
  mlir::Value dest = op->getOperand(1);
  mlir::ShapedType srcSTy = src.getType().dyn_cast<mlir::ShapedType>();
  mlir::ShapedType destSTy = dest.getType().dyn_cast<mlir::ShapedType>();

  // Get the required shape.
  llvm::SmallVector<mlir::Value> srcShape = llvm::SmallVector<mlir::Value>();
  llvm::SmallVector<mlir::Value> destShape =
      llvm::SmallVector<mlir::Value>(); // with the splatting dimensions
  FixUBUtils::getShapes(builder, loc, src, &srcShape);
  FixUBUtils::getShapes(builder, loc, dest, &destShape);

  // mactch the corresponding dimensions
  // to match the static and dynamic dimensions of the src type to the dest
  // type, but the two type are not exactly equal, needs to change the order of
  // the dest dimensions to the src dimensions order
  llvm::ArrayRef<int64_t> transposeParam =
      mlir::cast<mlir::linalg::TransposeOp>(*op).getPermutation();
  llvm::SmallVector<mlir::Value> toCompareShape =
      llvm::SmallVector<mlir::Value>(); // dimensions size same with dest, order
                                        // same with src
  llvm::SmallVector<int64_t>
      fixTargetShape; // dimensions size same with dest, order same with src
  // llvm::outs() << "destShapetmp size : " << destShapetmp.size() << "\n";
  for (int64_t i = 0; i < destShape.size(); i++) {
    // llvm::outs() << addingDim[i] << " ";
    int64_t transposeIdx = std::distance(
        transposeParam.begin(),
        std::find(transposeParam.begin(), transposeParam.end(), i));
    // llvm::outs() << transposeIdx << " ";
    toCompareShape.push_back(destShape[transposeIdx]);
    if (destSTy.getDimSize(transposeIdx) < 0)
      fixTargetShape.push_back(mlir::ShapedType::kDynamic);
    else
      fixTargetShape.push_back(destSTy.getDimSize(transposeIdx));
  }
  // llvm::outs() << "\n";

  // for (int64_t i = 0; i < toCompareShape.size(); i++) {
  //   llvm::outs() << toCompareShape[i] << " ";
  //   // destShape.erase(addingDim[i]);
  // }
  // llvm::outs() << "\n";
  // for (int64_t i = 0; i < fixTargetShape.size(); i++) {
  //   llvm::outs() << fixTargetShape[i] << " ";
  //   // destShape.erase(addingDim[i]);
  // }
  // llvm::outs() << "\n";

  // Compare the shape.
  mlir::Operation *sameOp =
      FixUBUtils::compareShape(builder, loc, srcShape, toCompareShape);
  mlir::Value sameShape = sameOp->getResult(0);

  // Write the dynamic check and fix code.
  auto sameBranch = [&](mlir::OpBuilder &b, mlir::Location loc) {
    // this operation restricts the input and output shape to have same static
    // dimensions, or both dynamic dimensions, so castToAllDynis unnecessary
    // mlir::Operation *newSrcOp =
    //     FixUBUtils::castToAllDyn(builder, loc, src);
    b.create<mlir::scf::YieldOp>(loc, src); // newSrcOp->getResult(0));
  };
  auto notSameBranch = [&](mlir::OpBuilder &b, mlir::Location loc) {
    // geting dynamic dims attributes
    llvm::SmallVector<mlir::Value> destDynDims =
        llvm::SmallVector<mlir::Value>();
    llvm::SmallVector<int64_t> destDynDimsIdices = llvm::SmallVector<int64_t>();
    FixUBUtils::getDynDims(builder, loc, dest, &destDynDims,
                           &destDynDimsIdices);

    // match the dynamic dimensions order same with the src type, add this dim
    // imformation in the attribute of memref.alloc/tensor.splat
    llvm::SmallVector<mlir::Value> dynDims = llvm::SmallVector<mlir::Value>();
    for (int64_t i = 0; i < srcShape.size(); i++) {

      // static dimensions
      if (fixTargetShape[i] > 0)
        continue;

      // llvm::outs() << i << "\n";
      // finding the matching index of the src dimension in the dest type
      int64_t transposeIdx = std::distance(
          transposeParam.begin(),
          std::find(transposeParam.begin(), transposeParam.end(), i));

      // llvm::outs() << transposeIdx << "\n";
      // llvm::outs() << destDynDimsIdices.size() << " " << destDynDimsIdices[0]
      // << "\n"; finding the matching index of the dynamic dimension(if it is)
      // in the extracted dyndims vector of the dest type
      int64_t dynDimIdx;
      llvm::SmallVector<int64_t>::iterator iter = std::find(
          destDynDimsIdices.begin(), destDynDimsIdices.end(), transposeIdx);
      if (!std::count(destDynDimsIdices.begin(), destDynDimsIdices.end(),
                      transposeIdx))
        continue;
      else
        dynDimIdx = std::distance(destDynDimsIdices.begin(), iter);

      // llvm::outs() << dynDimIdx << "\n";
      dynDims.push_back(destDynDims[dynDimIdx]);
    }

    // generating ub free dest shape
    mlir::Value newSrc;
    // llvm::outs() << "fix linalg.copy"
    if (llvm::isa<mlir::MemRefType>(src.getType())) {
      mlir::MemRefType newDestSTy =
          mlir::MemRefType::get(fixTargetShape, destSTy.getElementType());
      newSrc = builder.create<mlir::memref::AllocOp>(loc, newDestSTy, dynDims);

      mlir::Value randnumVal = FixUBUtils::genRandomIntOrFloatValue(
          builder, loc, newDestSTy.getElementType());
      builder.create<mlir::linalg::FillOp>(loc, randnumVal, newSrc);
    } else {
      mlir::RankedTensorType newDestSTy =
          mlir::RankedTensorType::get(fixTargetShape, destSTy.getElementType());
      mlir::Value randnumVal = FixUBUtils::genRandomIntOrFloatValue(
          builder, loc, destSTy.getElementType());
      newSrc = builder.create<mlir::tensor::SplatOp>(loc, randnumVal,
                                                     newDestSTy, dynDims);
    }

    // this operation restricts the input and output shape to have same static
    // dimensions, or both dynamic dimensions, so castToAllDynis unnecessary
    // mlir::Operation *newSrcOp =
    //     FixUBUtils::castToAllDyn(builder, loc, newSrc);
    b.create<mlir::scf::YieldOp>(loc, newSrc); // newSrcOp->getResult(0));
  };
  mlir::Operation *scfIfOp =
      builder.create<mlir::scf::IfOp>(loc, sameShape, sameBranch, notSameBranch)
          .getOperation();
  op->setOperand(0, scfIfOp->getResult(0));
  return;
}
