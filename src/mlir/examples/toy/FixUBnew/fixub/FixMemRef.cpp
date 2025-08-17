#include <string>

#include "llvm/Support/raw_ostream.h"

#include "mlir/Dialect/Affine/Passes.h"
// #include "mlir/Dialect/Index/IR/IndexEnums.h.inc"
#include "mlir/IR/Operation.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Parser/Parser.h"

#include "fixub/ConstantPool.h"
#include "fixub/DelOp.h"
#include "fixub/FixMemRef.h"
#include "fixub/FixUBUtils.h"

void FixMemref::fixmemref(mlir::Operation *op) {
  std::string opname = op->getName().getStringRef().str();
  if (opname == "memref.assume_alignment") {
    FixMemref::fixAssumeAlignment(op);
  }
  if (opname == "memref.copy") {
    FixMemref::fixCopy(op);
  }
  if (opname == "memref.realloc") {
    FixMemref::fixRealloc(op);
  }
  if (opname == "memref.alloc" || opname == "memref.alloca") {
    FixMemref::fixAlloc(op);
  }
  if (opname == "memref.cast") {
    FixMemref::fixCast(op);
  }
  if (opname == "memref.store") {
    FixMemref::fixStore(op);
  }
  if (opname == "memref.load") {
    FixMemref::fixLoad(op);
  }
  if (opname == "memref.dim") {
    FixMemref::fixDim(op);
  }
  return;
}

mlir::ModuleOp FixMemref::getParentModuleOp(mlir::Operation *op) {
  llvm::outs() << "[FixMemref::getParentModuleOp] "
               << op->getName().getStringRef().str() << "\n";
  while (op && !llvm::isa<mlir::ModuleOp>(op)) {
    op = op->getParentOp();
    llvm::outs() << "[FixMemref::getParentModuleOp] "
                 << op->getName().getStringRef().str() << "\n";
  }
  llvm::outs() << "[FixMemref::getParentModuleOp] "
               << op->getName().getStringRef().str() << "\n";
  return llvm::cast<mlir::ModuleOp>(op);
}

void FixMemref::fixAssumeAlignment(mlir::Operation *op) {
  /**
   * Fix memref.assume_alignment operation.
   * The undefined behaviors description is as follows:
   * The assume_alignment operation takes a memref and an integer of alignment
   * value, and internally annotates the buffer with the given alignment. If the
   * buffer isn’t aligned to the given alignment, the behavior is undefined.
   *
   * Example: memref.assume_alignment %0, 2 : memref<8x64xf32>
   *
   * This fix does the following things:
   * (1) Fix the alignment attribute of input memref
   */

  // Set environment
  llvm::outs() << "[FixMemref::fixAssumeAlignment]\n";
  mlir::Location loc = op->getLoc();
  mlir::OpBuilder builder(op->getBlock(), op->getBlock()->begin());
  builder.setInsertionPoint(op);

  // Get the operand
  // Get the alignment of memref operand,
  // if the memref creation operation has this attribute, we directly use this
  // attribute. if the memref creation operation does not has this attribute, we
  // use the default layout.
  mlir::Value memref = op->getOperand(0);
  mlir::Operation *memrefOp = memref.getDefiningOp();
  mlir::Attribute memrefAttr = NULL;
  if (memrefOp)
    memrefAttr = memrefOp->getAttr("alignment");
  int64_t memrefAlignAttrVal;
  llvm::outs() << "[this] debug point\n";
  if (!memrefAttr) {
    // This code comes from the unit test case, load the default alignment via
    // an empty file.
    mlir::MemRefType memrefType = memref.getType().cast<mlir::MemRefType>();
    const char *ir = R"MLIR(module {})MLIR";
    mlir::DialectRegistry registry;
    mlir::MLIRContext ctx(registry);
    mlir::OwningOpRef<mlir::ModuleOp> module =
        mlir::parseSourceString<mlir::ModuleOp>(ir, &ctx);
    mlir::DataLayout layout(module.get());
    memrefAlignAttrVal =
        layout.getTypeABIAlignment(memrefType.getElementType());
  } else {
    mlir::IntegerAttr memrefAlignmentAttr =
        memrefAttr.dyn_cast<mlir::IntegerAttr>();
    memrefAlignAttrVal = memrefAlignmentAttr.getInt();
  }
  // Get the alignment of this operation
  mlir::Attribute attr = op->getAttr("alignment");
  mlir::IntegerAttr alignmentAttr = attr.dyn_cast<mlir::IntegerAttr>();
  int64_t alignAttrVal = alignmentAttr.getInt();
  // Compare the aligment attribute
  if (memrefAlignAttrVal != alignAttrVal) {
    mlir::IntegerAttr newAlignAttr = mlir::IntegerAttr::get(
        mlir::IntegerType::get(op->getContext(), 32), memrefAlignAttrVal);
    op->setAttr("alignment", newAlignAttr);
  }
  return;
}

void FixMemref::fixCopy(mlir::Operation *op) {
  /**
   * Fix undefined behaviors in memref.copy.
   *
   * Example:
   * memref.copy %arg0, %arg1 : memref<?xf32> to memref<?xf32>
   *
   * The followinng inputs will be rejected by static semantic checking of
   * mlir-opt: (1) memrefs that have differnet ranks (2) memrefs taht have at
   * least one pair of static dimensions that are not same.
   *
   * The description in documentation is:
   * Source and destination are expected to have the same element type and
   * shape. Otherwise, the result is undefined. They may have different layouts.
   *
   * Note that: the element type is checked during static semantic checking.
   *
   * This fix funtion writet the dynamic check and fix code to ensuring the
   * shape of src memref and dest memref.
   */
  // Set environment
  mlir::Location loc = op->getLoc();
  mlir::OpBuilder builder(op->getBlock(), op->getBlock()->begin());
  builder.setInsertionPoint(op);

  // Get the Operand
  mlir::Value srcMemref = op->getOperand(0);
  mlir::Value destMemref = op->getOperand(1);
  mlir::ShapedType srcMemrefSTy =
      srcMemref.getType().dyn_cast<mlir::ShapedType>();
  mlir::ShapedType destMemrefSTy =
      destMemref.getType().dyn_cast<mlir::ShapedType>();
  mlir::MemRefType srcMemrefTy =
      srcMemref.getType().dyn_cast<mlir::MemRefType>();
  mlir::MemRefType destMemrefTy =
      destMemref.getType().dyn_cast<mlir::MemRefType>();

  // Get the required shape.
  llvm::SmallVector<mlir::Value> srcMemrefShape =
      llvm::SmallVector<mlir::Value>();
  llvm::SmallVector<mlir::Value> destMemrefShape =
      llvm::SmallVector<mlir::Value>();
  FixUBUtils::getShapes(builder, loc, srcMemref, &srcMemrefShape);
  FixUBUtils::getShapes(builder, loc, destMemref, &destMemrefShape);

  // Compare the shape.
  mlir::Operation *sameOp =
      FixUBUtils::compareShape(builder, loc, srcMemrefShape, destMemrefShape);
  mlir::Value sameShape = sameOp->getResult(0);

  // Write the dynamic check and fix code.
  auto sameBranch = [&](mlir::OpBuilder &b, mlir::Location loc) {
    mlir::Operation *newSrcOp =
        FixUBUtils::castToAllDyn(builder, loc, srcMemref);
    b.create<mlir::scf::YieldOp>(loc, newSrcOp->getResult(0));
  };
  auto notSameBranch = [&](mlir::OpBuilder &b, mlir::Location loc) {
    llvm::SmallVector<mlir::Value> dynDims = llvm::SmallVector<mlir::Value>();
    llvm::SmallVector<int64_t> dynDimsIdx =
        llvm::SmallVector<int64_t>(); // this is unused here because the src and
                                      // dest shape should be strictly same
    FixUBUtils::getDynDims(builder, loc, destMemref, &dynDims, &dynDimsIdx);
    mlir::Value newSrc =
        builder.create<mlir::memref::AllocOp>(loc, destMemrefTy, dynDims);

    mlir::Value randnumVal = FixUBUtils::genRandomIntOrFloatValue(
        builder, loc, destMemrefTy.getElementType());
    builder.create<mlir::linalg::FillOp>(loc, randnumVal, newSrc);

    mlir::Operation *newSrcOp = FixUBUtils::castToAllDyn(builder, loc, newSrc);
    b.create<mlir::scf::YieldOp>(loc, newSrcOp->getResult(0));
  };
  mlir::Operation *scfIfOp =
      builder.create<mlir::scf::IfOp>(loc, sameShape, sameBranch, notSameBranch)
          .getOperation();
  op->setOperand(0, scfIfOp->getResult(0));
  return;
}

void FixMemref::fullMemrefWithRandVal(mlir::OpBuilder builder,
                                      mlir::Location loc, mlir::Value memref,
                                      mlir::Value lowerBound,
                                      mlir::Value upperBound) {
  /**
   * Full the given memref value with a serious random values, from the
   * lowerBound(include) to the upperBound(not include).
   *
   * Params:
   * mlir::OpBuilder builder, to support MLIR-API calls
   * mlir::Location loc, to support MLIR-API calls
   * mlir::Value memref, the value to be fulled.
   * mlir::Value lowerBound, the start index.
   * mlir::Value upperBound, the end index.
   *
   * Return:
   * None, because the memref.store operation does not have any result.
   */
  llvm::outs() << "[FixMemref::fullMemrefWithRandVal]\n";

  // Generate a random init seed. The current version considers the integer
  // only.
  mlir::ShapedType memrefSTy = memref.getType().dyn_cast<mlir::ShapedType>();
  mlir::Type elemTy = memrefSTy.getElementType();
  if (elemTy.isIntOrIndex()) {
    // Calculate the seed
    uint64_t seed = FixUBUtils::randuval(64);

    llvm::outs() << "the point before\n";

    mlir::Value seedVal = ConstantPool::getValue(seed, builder.getI64Type());

    llvm::outs() << "the point after\n";
    // Use scf for to full the given memref.
    // Calculate some numbers that used in random number calculate.
    // Use arith.trunci to ensure the correctness of the calculation.
    // The support type is no more than i64
    // #define MULTIPLIER 1103515245
    // #define INCREMENT 12345
    // #define MODULUS 2147483648 // 2^31
    int64_t __MULTIPLIER__ =
        0X41c64e6d41c64e6d; // 0X41c64e6d41c64e6d, binary repeat of 1103515245
    int64_t __INCREMENT__ =
        0X0000303900003039; // 0X0000303900003039, binary repeat of 12345,
    int64_t __MODULUS__ = 0X8000000000000000; // 2 ^ 63

    mlir::Value MULTIPLIER_VAL =
        ConstantPool::getValue(__MULTIPLIER__, builder.getI64Type());
    mlir::Value INCREMENT_VAL =
        ConstantPool::getValue(__INCREMENT__, builder.getI64Type());
    mlir::Value MODULUS_VAL =
        ConstantPool::getValue(__MODULUS__, builder.getI64Type());

    llvm::SmallVector<mlir::Value> iterArgs = llvm::SmallVector<mlir::Value>();
    iterArgs.push_back(seedVal);
    auto blockBuilder = [&](mlir::OpBuilder &builder, mlir::Location loc,
                            mlir::Value iv /*loop iterator*/,
                            mlir::ValueRange args) {
      mlir::Value seed = args[0];
      // Calculate the new element using given formular, and convert it to index
      // or integer with specific bitwidth. seed = (MULTIPLIER * seed +
      // INCREMENT) % MODULUS;
      seed = builder.create<mlir::arith::MulIOp>(loc, seed, MULTIPLIER_VAL)
                 .getResult();
      seed = builder.create<mlir::arith::AddIOp>(loc, seed, INCREMENT_VAL)
                 .getResult();
      seed = builder.create<mlir::arith::RemSIOp>(loc, seed, MODULUS_VAL)
                 .getResult();
      mlir::Value newElem;
      if (elemTy.isIndex()) { // the element is index
        newElem = builder
                      .create<mlir::arith::IndexCastOp>(
                          loc, builder.getIndexType(), seed)
                      .getResult();
      } else {
        // assume that the element is integer
        newElem = builder.create<mlir::arith::TruncIOp>(loc, elemTy, seed)
                      .getResult();
      }
      // insert this value into memref, and return the iteration arguments.
      builder.create<mlir::memref::StoreOp>(loc, newElem, memref, iv);
      llvm::SmallVector<mlir::Value> iterArgs =
          llvm::SmallVector<mlir::Value>();
      iterArgs.push_back(seed);
      builder.create<mlir::scf::YieldOp>(loc, iterArgs);
    };
    mlir::Operation *scfForOp = builder.create<mlir::scf::ForOp>(
        loc, lowerBound, upperBound,
        ConstantPool::getValue(1, builder.getIndexType()), iterArgs,
        blockBuilder);
  }
  return;
}

void FixMemref::fixRealloc(mlir::Operation *op) {
  /**
   * This function fix memref.realloc operation.
   *
   * Example:
   * %0 = memref.realloc %src : memref<64xf32> to memref<124xf32>
   *
   * This fix work does the following tasks:
   * (1) Fix the use of old SSA value.
   * (2) Fix the expand content(because the value content of the expanded memref
   * is undefined).
   */

  // Set environment
  mlir::Location loc = op->getLoc();
  mlir::OpBuilder builder(op->getBlock(), op->getBlock()->begin());
  builder.setInsertionPointAfter(op);

  // Get Operand(and, interestingly, the result)
  mlir::Value oldMemref = op->getOperand(0);
  mlir::Value newMemref = op->getResult(0);

  // Fix the use after this opeartion.
  llvm::SmallVector<mlir::Operation *> userList =
      llvm::SmallVector<mlir::Operation *>();
  for (mlir::Operation *user : oldMemref.getUsers()) {
    userList.push_back(user);
  }
  bool init = false;
  mlir::Value newOperand;
  for (mlir::Operation *user : userList) {
    int64_t useOpLine = FixUBUtils::getLineNumber(user);
    int64_t thisOpLine = FixUBUtils::getLineNumber(op);
    if (useOpLine > thisOpLine) {
      int64_t targetOpIdx = 0;
      int64_t userOperandNum = user->getNumOperands();
      while (targetOpIdx < userOperandNum &&
             oldMemref != user->getOperand(targetOpIdx)) {
        targetOpIdx++;
      }
      if (!init) {
        // Old Comments:
        // Cast the new operand to dynamic shape, and cast the dynamic shape to
        // the original result type. Because that the old use should have one
        // type, one suitable operand is enough.

        // New comments:
        // Generate a new memref with the original one.

        mlir::Value oldOprand = user->getOperand(targetOpIdx);
        mlir::MemRefType oldOprandMemrefTy =
            oldOprand.getType().dyn_cast<mlir::MemRefType>();

        builder.setInsertionPointAfter(
            FixUBUtils::getDifiningOpIncludeArgs(builder, loc, oldOprand));

        llvm::SmallVector<mlir::Value> dynDims =
            llvm::SmallVector<mlir::Value>();
        FixUBUtils::getDynDims(builder, loc, oldOprand, &dynDims);

        newOperand =
            builder
                .create<mlir::memref::AllocOp>(loc, oldOprandMemrefTy, dynDims)
                .getResult();

        mlir::Value randnumVal = FixUBUtils::genRandomIntOrFloatValue(
            builder, loc, oldOprandMemrefTy.getElementType());
        builder.create<mlir::linalg::FillOp>(loc, randnumVal, newOperand);

        // llvm::SmallVector<int64_t> tmpShape = llvm::SmallVector<int64_t>();
        // tmpShape.push_back(mlir::ShapedType::kDynamic);
        // mlir::Value tmpOperand = builder.create<mlir::memref::CastOp>(
        //     loc,
        //     mlir::MemRefType::get(tmpShape,
        //     oldOprandMemrefTy.getElementType()), newMemref);
        // newOperand = builder.create<mlir::memref::CastOp>(
        //     loc, oldOprandMemrefTy, tmpOperand);

        builder.setInsertionPointAfter(op);
        init = true;
      }
      assert(targetOpIdx < userOperandNum);
      user->setOperand(targetOpIdx, newOperand);
    }
  }

  // Init the remain data of the new memref.
  builder.setInsertionPointAfter(op);
  llvm::SmallVector<mlir::Value> oldShape = llvm::SmallVector<mlir::Value>();
  llvm::SmallVector<mlir::Value> newShape = llvm::SmallVector<mlir::Value>();
  FixUBUtils::getShapes(builder, loc, oldMemref, &oldShape);
  FixUBUtils::getShapes(builder, loc, newMemref, &newShape);
  mlir::Value oldSize = oldShape[0];
  mlir::Value newSize = newShape[0];
  // Calculate the range of new memref that should be fulled.
  mlir::Value expand =
      builder
          .create<mlir::index::CmpOp>(loc, mlir::index::IndexCmpPredicate::UGT,
                                      newSize, oldSize)
          .getResult();
  auto expandBuilder = [&](mlir::OpBuilder &b, mlir::Location loc) {
    FixMemref::fullMemrefWithRandVal(builder, loc, newMemref, oldSize, newSize);
    builder.create<mlir::scf::YieldOp>(loc);
  };
  // auto notExpandBuilder = [&](mlir::OpBuilder &b, mlir::Location loc) { };
  builder.create<mlir::scf::IfOp>(loc, expand, expandBuilder);
  return;
}

void FixMemref::fixAlloc(mlir::Operation *op) {
  /**
   * Fix undefined behaviors in memref.alloc and memref.alloca.
   *
   * Example:
   * %alloc = memref.alloc(%2, %5) : memref<?x?xi64>
   *
   * This fix work does the following tasks:
   * (1) Fix the mle problem caused by too large dynamic dimensions.
   * (2) Fix the uninitialized memref generated by memref.alloca(the
   * fixing algorithm initialize the memref.alloc as well).
   *
   *
   * This fix funtion writet the dynamic check and fix code to ensuring the
   * dynamic dimensions will not cause mle problems, and initialize the memref
   * value with random value.
   *
   * Modified at 2024.11.05:
   * add logic to replace alloca with alloc in loops.
   */
  // Set environment
  mlir::Location loc = op->getLoc();
  mlir::OpBuilder builder(op->getBlock(), op->getBlock()->begin());
  builder.setInsertionPoint(op);

  // Get the Operand
  mlir::Value destMemref = op->getResult(0);
  mlir::ShapedType destMemrefSTy =
      destMemref.getType().dyn_cast<mlir::ShapedType>();
  mlir::MemRefType destMemrefTy =
      destMemref.getType().dyn_cast<mlir::MemRefType>();
  std::string opname = op->getName().getStringRef().str();
  llvm::SmallVector<mlir::Value> dynDimsIdices;
  if (opname == "memref.alloc")
    dynDimsIdices = mlir::cast<mlir::memref::AllocOp>(*op).getDynamicSizes();
  else
    dynDimsIdices = mlir::cast<mlir::memref::AllocaOp>(*op).getDynamicSizes();

  // for(int64_t i = 0; i < dynDimsIdices.size(); i++) {
  //   llvm::outs() << dynDimsIdices[i] << " ";
  // }
  // llvm::outs() << "\n";

  // generate the maxsize and 1 in sub 1 const values
  llvm::SmallVector<mlir::Value> fixedDynDimsIdices;
  mlir::Value topSize = ConstantPool::getValue(20, builder.getIndexType());
  mlir::Value num1 = ConstantPool::getValue(1, builder.getIndexType());

  // compare and fix every dynamic dimensions
  for (int64_t i = 0; i < dynDimsIdices.size(); i++) {

    // to avoid dimsize as 0, sub 1 first
    mlir::Value minus1Val =
        builder.create<mlir::index::SubOp>(loc, dynDimsIdices[i], num1)
            .getResult();
    mlir::Value cmpVal =
        builder
            .create<mlir::index::CmpOp>(
                loc, mlir::index::IndexCmpPredicate::ULE, minus1Val, topSize)
            .getResult();

    // Write the dynamic check and fix code.
    auto sameBranch = [&](mlir::OpBuilder &b, mlir::Location loc) {
      b.create<mlir::scf::YieldOp>(loc, dynDimsIdices[i]);
    };
    auto notSameBranch = [&](mlir::OpBuilder &b, mlir::Location loc) {
      // generate new dynamic dim size
      int64_t randNum = FixUBUtils::randsval(1, 15);
      mlir::Value randConstVal =
          builder.create<mlir::index::ConstantOp>(loc, randNum).getResult();

      b.create<mlir::scf::YieldOp>(loc, randConstVal);
    };
    mlir::Operation *scfIfOp =
        builder.create<mlir::scf::IfOp>(loc, cmpVal, sameBranch, notSameBranch)
            .getOperation();
    // llvm::outs() << scfIfOp->getResult(0) << "\n";
    fixedDynDimsIdices.push_back(scfIfOp->getResult(0));
  }
  // llvm::outs() << destMemrefTy << "\n";
  // for(int64_t i = 0; i < fixedDynDimsIdices.size(); i++) {
  //   llvm::outs() << fixedDynDimsIdices[i] << " ";
  // }
  // llvm::outs() << "\n";

  mlir::Operation *loopOp = op->getParentOp();
  bool in_loop = false;
  while (loopOp) {
    std::string opParams = FixUBUtils::getOperationNameParamType(loopOp);
    in_loop = opParams.find("linalg") != std::string::npos ||
              opParams.find("affine.for") != std::string::npos ||
              opParams.find("scf.for") != std::string::npos;
    if (in_loop)
      break;
    loopOp = loopOp->getParentOp();
  }

  mlir::Operation *newSrc;
  if (opname == "memref.alloc" || in_loop)
    newSrc = builder
                 .create<mlir::memref::AllocOp>(loc, destMemrefTy,
                                                fixedDynDimsIdices)
                 .getOperation();
  else
    newSrc = builder
                 .create<mlir::memref::AllocaOp>(loc, destMemrefTy,
                                                 fixedDynDimsIdices)
                 .getOperation();

  op->replaceAllUsesWith(newSrc);

  // initialize the memref value of a const value with linalg.fill
  mlir::Value randnumVal = FixUBUtils::genRandomIntOrFloatValue(
      builder, loc, destMemrefTy.getElementType());
  builder.create<mlir::linalg::FillOp>(loc, randnumVal, newSrc->getResult(0));

  delOps.push_back(op);

  return;
}

void FixMemref::fixCast(mlir::Operation *op) {
  /**
   * Fix undefined behaviors in memref.cast.
   *
   * Example:
   * %cast = memref.cast %5 : memref<?x?x?xi64> to memref<15x?x16xi64>
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
  mlir::Value destMemref = op->getResult(0);
  mlir::ShapedType destMemrefSTy =
      destMemref.getType().dyn_cast<mlir::ShapedType>();
  mlir::MemRefType destMemrefTy =
      destMemref.getType().dyn_cast<mlir::MemRefType>();

  mlir::Value srcMemref = op->getOperand(0);
  mlir::ShapedType srcMemrefSTy =
      srcMemref.getType().dyn_cast<mlir::ShapedType>();
  mlir::MemRefType srcMemrefTy =
      srcMemref.getType().dyn_cast<mlir::MemRefType>();
  // // for(int64_t i = 0; i < dynDimsIdices.size(); i++) {
  // //   llvm::outs() << dynDimsIdices[i] << " ";
  // // }
  // // llvm::outs() << "\n";

  // for ever tocompare dim pairs, check the consistence, form the dynamic
  // attribute for memref.alloc
  llvm::SmallVector<mlir::Value> fixedDynDimsIdices;
  mlir::Value shapeSame = ConstantPool::getValue(1, builder.getI1Type());
  bool needToCompare = false;
  for (int64_t i = 0; i < srcMemrefSTy.getRank(); i++) {
    // input dim is static
    if (srcMemrefSTy.getDimSize(i) > 0)
      continue;
    mlir::Value curDim = ConstantPool::getValue(i, builder.getIndexType());
    mlir::Value curInputSize =
        builder.create<mlir::memref::DimOp>(loc, srcMemref, curDim).getResult();
    // both input and output dim are dynamic
    if (destMemrefSTy.getDimSize(i) < 0) {
      fixedDynDimsIdices.push_back(curInputSize);
      continue;
    }
    needToCompare = true;
    // use the num same with the static dim as the attribute of the new dyn dim
    mlir::Value curOutputSize = ConstantPool::getValue(
        destMemrefSTy.getDimSize(i), builder.getIndexType());
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
      b.create<mlir::scf::YieldOp>(loc, srcMemref);
    };
    auto notSameBranch = [&](mlir::OpBuilder &b, mlir::Location loc) {
      // generate new dynamic dim size to cast memref
      mlir::Value newSrcRetVal = builder
                                     .create<mlir::memref::AllocOp>(
                                         loc, srcMemrefTy, fixedDynDimsIdices)
                                     .getResult();

      mlir::Value randnumVal = FixUBUtils::genRandomIntOrFloatValue(
          builder, loc, srcMemrefTy.getElementType());
      builder.create<mlir::linalg::FillOp>(loc, randnumVal, newSrcRetVal);

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

void FixMemref::fixStore(mlir::Operation *op) {
  /**
   * Fix undefined behaviors in memref.store.
   *
   * Example:
   * memref.store %val, %mem[%idx1, %idx2] : memref<4x?xf32>
   *
   * This fix work does the following tasks:
   * (1) Ensure the indices %idx1 and %idx2 valid.
   */
  // Set environment
  mlir::Location loc = op->getLoc();
  mlir::OpBuilder builder(op->getBlock(), op->getBlock()->begin());
  builder.setInsertionPoint(op);

  // Get Operand
  mlir::Value memVal = op->getOperand(1);
  mlir::ShapedType memValTy =
      mlir::dyn_cast<mlir::ShapedType>(memVal.getType());
  llvm::SmallVector<mlir::Value> indices = llvm::SmallVector<mlir::Value>();
  for (int64_t i = 2; i < op->getNumOperands(); ++i) {
    indices.push_back(op->getOperand(i));
  }

  // Get shapes
  llvm::SmallVector<mlir::Value> memShape = llvm::SmallVector<mlir::Value>();
  FixUBUtils::getShapes(builder, loc, memVal, &memShape);

  // Generate new indices
  llvm::SmallVector<mlir::Value> newIndices = llvm::SmallVector<mlir::Value>();
  FixUBUtils::getValidIndex(builder, loc, memShape, indices, &newIndices);

  // Set operands
  for (int64_t i = 2; i < op->getNumOperands(); ++i) {
    op->setOperand(i, newIndices[i - 2]);
  }

  return;
}

void FixMemref::fixLoad(mlir::Operation *op) {
  /**
   * Fix undefined behaviors in memref.cast.
   *
   * Example:
   * %12 = memref.load %mem[%idx1, %idx2] : memref<8x?xi32, #layout, memspace0>
   *
   * This fix work does the following tasks:
   * (1) Ensure the indices %idx1 and %idx2 valid.
   */
  // Set environment
  mlir::Location loc = op->getLoc();
  mlir::OpBuilder builder(op->getBlock(), op->getBlock()->begin());
  builder.setInsertionPoint(op);

  // Get Operand
  mlir::Value memVal = op->getOperand(0);
  mlir::ShapedType memValTy =
      mlir::dyn_cast<mlir::ShapedType>(memVal.getType());
  llvm::SmallVector<mlir::Value> indices = llvm::SmallVector<mlir::Value>();
  for (int64_t i = 1; i < op->getNumOperands(); ++i) {
    indices.push_back(op->getOperand(i));
  }

  // Get shapes
  llvm::SmallVector<mlir::Value> memShape = llvm::SmallVector<mlir::Value>();
  FixUBUtils::getShapes(builder, loc, memVal, &memShape);

  // Generate new indices
  llvm::SmallVector<mlir::Value> newIndices = llvm::SmallVector<mlir::Value>();
  FixUBUtils::getValidIndex(builder, loc, memShape, indices, &newIndices);

  // Set operands
  for (int64_t i = 1; i < op->getNumOperands(); ++i) {
    op->setOperand(i, newIndices[i - 1]);
  }

  return;
}

void FixMemref::fixDim(mlir::Operation *op) {
  /**
   * This function fixes undefined behavior in memref.dim.
   * The documentation about UB says:
   * If the dimension index is out of bounds, the behavior is undefined.
   *
   * Consider the following example:
   * %dim_size = memref.dim %memref, %dim : memref<4x?xf32>
   * If %dim is higher than 2, then the undefined behavoir occurs.
   *
   * For fix this, %dim should compare to the rank of %memref,
   * and do %dim = %dim % rank(%memref)
   */
  mlir::Location loc = op->getLoc();
  mlir::OpBuilder builder(op->getBlock(), op->getBlock()->begin());
  builder.setInsertionPoint(op);

  mlir::Value mem = op->getOperand(0), dim = op->getOperand(1);

  auto elementTy = llvm::cast<mlir::ShapedType>(mem.getType()).getElementType();

  // FlaotType should not be banded, because memref.dim got index type value
  // if(llvm::isa<mlir::FloatType>(elementTy))
  //     return ;

  auto rank = builder.create<mlir::memref::RankOp>(loc, mem);
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
  return;
}