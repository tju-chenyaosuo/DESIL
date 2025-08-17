#include <random>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"

#include "fixub/ConstantPool.h"
#include "fixub/FixUBUtils.h"

#include "mlir/IR/Operation.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Parser/Parser.h"

std::string FixUBUtils::getValueTypeStr(mlir::Value v) { // viewed
  mlir::Type vTy = v.getType();

  llvm::SmallVector<char, 128> buffer;
  llvm::raw_svector_ostream stream(buffer);
  vTy.print(stream);

  std::string ty = stream.str().str();
  return ty;
}

mlir::IntegerAttr FixUBUtils::getI(mlir::Value value) { // viewed
  /**
   * Get the value of opeartion that has "value" attribute, such as
   * arith.constant.
   */
  auto type = mlir::dyn_cast<mlir::IntegerType>(value.getType());
  if (type) {
    auto attr =
        value.getDefiningOp()->getAttrOfType<mlir::IntegerAttr>("value");
    if (attr) {
      // auto integerValue = attr.getInt();
      return attr;
    }
  }
}

std::int64_t FixUBUtils::randsval(int64_t bitwidth) { // viewed
  // llvm::outs() << min1 << " " << max1 << "\n";
  std::random_device rd1;
  std::mt19937 gen1(rd1());

  std::uniform_int_distribution<int64_t> dis1(1, bitwidth);
  int64_t randomBitwidth = dis1(gen1);

  int64_t max1 = FixUBUtils::bitsmax(randomBitwidth),
          min1 = FixUBUtils::bitsmin(randomBitwidth);
  // llvm::outs()<<randomBitwidth << " max: " << max1 << " min: " << min1 <<
  // "\n";
  std::random_device rd;
  std::mt19937 gen(rd());

  std::uniform_int_distribution<int64_t> dis(min1, max1);
  int64_t randomValue = dis(gen);
  return randomValue;
}

std::int64_t FixUBUtils::randsval(int64_t lb, int64_t ub) { // viewed
  std::random_device rd1;
  std::mt19937 gen1(rd1());

  std::uniform_int_distribution<int64_t> dis1(lb, ub);
  int64_t randval = dis1(gen1);

  return randval;
}

std::uint64_t FixUBUtils::randuval(uint64_t bitwidth) { // viewed
  // 1. Generate a random bitwidth
  // llvm::outs() << min1 << " " << max1 << "\n";
  std::random_device rd1;
  std::mt19937 gen1(rd1());
  std::uniform_int_distribution<uint64_t> dis1(1, bitwidth);
  uint64_t randomBitwidth = dis1(gen1);

  // 2. Get the bound of random number, according to the bitwidth.
  // uint64_t max1=FixUBUtils::bitumax(randomBitwidth),min1=0;
  uint64_t max1 = FixUBUtils::bitumax(randomBitwidth), min1 = 0;
  // llvm::outs()<<randomBitwidth << " max: " << max1 << " min: " << min1 <<
  // "\n";
  std::random_device rd;
  std::mt19937 gen(rd());

  // 3. Generate the number
  std::uniform_int_distribution<uint64_t> dis(min1, max1);
  uint64_t randomValue = dis(gen);
  return randomValue;
}

std::uint64_t FixUBUtils::bitumax(uint64_t bitwidth) { // viewd
  if (bitwidth == 1)
    return 1;
  if (bitwidth == 64)
    return (((1ULL << 63) - 1) << 1) + 1;
  else
    return pow(2, bitwidth) - 1;
}

std::int64_t FixUBUtils::bitsmax(int64_t bitwidth) { // viewd
  if (bitwidth == 1)
    return 1;
  if (bitwidth == 64) // why not directly use "1ULL<<(bitwidth-1))-1" for all of
                      // bitwidth except 1?
    return (int64_t)((1ULL << (bitwidth - 1)) - 1);
  else
    return pow(2, bitwidth - 1) - 1;
}

std::int64_t FixUBUtils::bitsmin(int64_t bitwidth) { // viewed
  if (bitwidth == 1)
    return 0;
  if (bitwidth == 64)
    return (int64_t)(1ULL << (bitwidth - 1));
  else
    return -pow(2, bitwidth - 1);
}

std::string FixUBUtils::getValueTypeStr(mlir::Type vTy) {
  llvm::SmallVector<char, 128> buffer;
  llvm::raw_svector_ostream stream(buffer);
  vTy.print(stream);

  std::string ty = stream.str().str();

  return ty;
}

std::string FixUBUtils::getOperationNameParamType(mlir::Operation *o) {
  std::string opName = "Unknown";
  std::string paramTys = "";
  if (o) {
    opName = o->getName().getStringRef().str();
    if (o->getNumOperands() != 0) {
      paramTys = getValueTypeStr(o->getOperand(0));
    }
    for (unsigned operandIdx = 1; operandIdx < o->getNumOperands();
         operandIdx++) {
      std::string paramTy = getValueTypeStr(o->getOperand(operandIdx));
      paramTys = paramTys + ", " + paramTy;
    }
  }
  std::string s = opName + "(" + paramTys + ")";
  return s;
}

std::string FixUBUtils::getLocationStr(mlir::Operation *o) {
  mlir::Location loc = o->getLoc();

  llvm::SmallVector<char, 128> buffer;
  llvm::raw_svector_ostream stream(buffer);
  loc.print(stream);

  std::string locStr = stream.str().str();
  return locStr;
}

std::int64_t FixUBUtils::getLineNumber(mlir::Operation *o) {
  // loc("test.mlir":1:1)
  if (!o)
    return 0;
  std::string pos = getLocationStr(o);
  int64_t end = pos.find_last_of(":");
  int64_t start = pos.find_first_of(":");
  if (end == std::string::npos || start == std::string::npos)
    return getLineNumber(o->getParentOp());
  int64_t strLen = end - start - 1;
  std::string line = pos.substr(start + 1, strLen);
  return std::stoi(line);
}

bool FixUBUtils::isDynShape(mlir::ShapedType sTy) {
  for (int64_t i = 0; i < sTy.getRank(); i++) {
    if (sTy.isDynamic(i) || sTy.getDimSize(i) <= 0)
      return true;
  }
  return false;
}

mlir::Value FixUBUtils::genRandomTensorFromExisting(mlir::ShapedType tensorTy,
                                                    mlir::OpBuilder builder,
                                                    mlir::Location loc) {
  // get new dimensions
  llvm::SmallVector<int64_t> newDims;
  bool isDynamicDim = false;
  for (int64_t i = 0; i < tensorTy.getRank(); i++) {
    // it is strange that isDynamic() is always return true, even if the
    // corresponding dimension is ``?''. I add the second condition to recognize
    // the dynamic size. tensorTy.getDimSize() always returns
    // -9223372036854775808 for dynamic dimension.
    if (tensorTy.isDynamic(i) ||
        tensorTy.getDimSize(i) <=
            0) { // is dynamic dimension or invalid dimension.
      int64_t randDim = FixUBUtils::randsval(1, 5);
      newDims.push_back(randDim);
      isDynamicDim = true;
    } else {
      int64_t randDim = tensorTy.getDimSize(i);
      newDims.push_back(randDim);
    }
  }
  // create another operation, and cast to the original shape.
  int64_t total_elem = 1;
  mlir::Operation *newValOp;
  mlir::Type elemTy = tensorTy.getElementType();
  for (auto dim : newDims) {
    total_elem *= dim;
  }

  if (total_elem >= 50) {
    int64_t randVal = FixUBUtils::randsval(elemTy.getIntOrFloatBitWidth());

    mlir::Value elemVal;
    if (mlir::isa<mlir::FloatType>(elemTy)) {
      elemVal = builder
                    .create<mlir::arith::ConstantOp>(
                        loc, mlir::FloatAttr::get(elemTy, 1.0))
                    .getOperation()
                    ->getResult(0);
    }
    if (mlir::isa<mlir::IntegerType>(elemTy) ||
        mlir::isa<mlir::IndexType>(elemTy)) {
      elemVal = builder
                    .create<mlir::arith::ConstantOp>(
                        loc, mlir::IntegerAttr::get(elemTy, randVal))
                    .getOperation()
                    ->getResult(0);
    }
    mlir::RankedTensorType newTensorTy =
        mlir::RankedTensorType::get(newDims, elemTy);
    newValOp = builder.create<mlir::tensor::SplatOp>(loc, elemVal, newTensorTy)
                   .getOperation();
  } else {
    llvm::SmallVector<mlir::Value> elemVals;
    for (int64_t i = 0; i < total_elem; i++) {
      int64_t randVal = FixUBUtils::randsval(elemTy.getIntOrFloatBitWidth());
      mlir::Value elemVal;

      if (mlir::isa<mlir::FloatType>(elemTy)) {
        elemVal = builder
                      .create<mlir::arith::ConstantOp>(
                          loc, mlir::FloatAttr::get(elemTy, 1.0))
                      .getOperation()
                      ->getResult(0);
      }
      if (mlir::isa<mlir::IntegerType>(elemTy) ||
          mlir::isa<mlir::IndexType>(elemTy)) {
        elemVal = builder
                      .create<mlir::arith::ConstantOp>(
                          loc, mlir::IntegerAttr::get(elemTy, randVal))
                      .getOperation()
                      ->getResult(0);
      }

      elemVals.push_back(elemVal);
    }
    mlir::RankedTensorType newTensorTy =
        mlir::RankedTensorType::get(newDims, elemTy);
    newValOp =
        builder.create<mlir::tensor::FromElementsOp>(loc, newTensorTy, elemVals)
            .getOperation();
  }
  assert(newValOp);
  // cast to the original (dynamic) shape
  if (isDynamicDim) {
    newValOp =
        builder
            .create<mlir::tensor::CastOp>(loc, tensorTy, newValOp->getResult(0))
            .getOperation();
  }
  return newValOp->getResult(0);
}

mlir::Value FixUBUtils::genRandomIntOrFloatValue(mlir::OpBuilder builder,
                                                 mlir::Location loc,
                                                 mlir::Type type) {
  /**
   * Generate a constant int or float random number
   *
   * Param:
   * mlir::OpBuilder builder, to support MLIR-API calls.
   * mlir::Location loc, to support MLIR-API calls.
   * mlir::Type type, the value type.
   *
   * Return:
   * mlir::Value : the value of the random number
   *
   * for float type ,for it's not included in checksum, just returns a constant
   * value of "1.0"
   */
  if (llvm::isa<mlir::FloatType>(type)) {
    return builder.create<mlir::arith::ConstantOp>(
        loc, mlir::FloatAttr::get(type, 1));
  }
  int64_t bitwidth = type.getIntOrFloatBitWidth();
  int64_t randnum = FixUBUtils::randsval(bitwidth);
  mlir::Value randnumVal = builder
                               .create<mlir::arith::ConstantOp>(
                                   loc, mlir::IntegerAttr::get(type, randnum))
                               .getOperation()
                               ->getResult(0);
  return randnumVal;
}

// changed params:
// the usage of this function is to list the dynamic dimensions for the
// attributes in the new memref.malloc/tensor.splat in ubfixing however, the
// existence and position of these dimensions in the before fixing dest type are
// not strictly same as the to fixed src target types e.g. %0 = linalg.matmul
// ins(%cast, %splat_0 : tensor<?x9xi32>, tensor<9x?xi32>) outs(%cast_2 :
// tensor<?x?xi32>) -> tensor<?x?xi32> the ?xs in the old dest type
// tensor<?x?xi32> correspond to the first dimension of the to fix src1 type and
// the second dimension of the to fix src2 type

// the old function just simplily extracted the dimensions, and the origin
// postion imformation(of what their origin indices in the old dest type are)
// are lost the change is to add a idx vector, to record the origin indices of
// these dynamic dimensions in the dest type

void FixUBUtils::getDynDims(mlir::OpBuilder builder, mlir::Location loc,
                            mlir::Value val,
                            llvm::SmallVector<mlir::Value> *shapes,
                            llvm::SmallVector<int64_t> *idx) {
  /**
   * Get the size of dynamic dims (memref or tensor)
   *
   * Param:
   * mlir::OpBuilder builder, to support MLIR-API calls.
   * mlir::Location loc, to support MLIR-API calls.
   * mlir::Value val, the value to be calculated.
   * llvm::SmallVector<mlir::Value>* shapes, the llvm::SmallVector that stores
   * the sizes of dynamic dimensions.
   *
   * Return:
   * None
   */
  mlir::ShapedType sTy = val.getType().dyn_cast<mlir::ShapedType>();
  llvm::ArrayRef<long int> shape = sTy.getShape();
  for (int64_t i = 0; i < shape.size(); i++) {
    if (shape[i] < 0) {
      // use memref.dim to get the corresponding size
      mlir::Value curDim = ConstantPool::getValue(i, builder.getIndexType());
      mlir::Value curSize;
      if (llvm::isa<mlir::MemRefType>(val.getType()))
        curSize =
            builder.create<mlir::memref::DimOp>(loc, val, curDim).getResult();
      else
        curSize =
            builder.create<mlir::tensor::DimOp>(loc, val, curDim).getResult();
      shapes->push_back(curSize);
      idx->push_back(i);
    }
  }
  return;
}

void FixUBUtils::getDynDims(mlir::OpBuilder builder, mlir::Location loc,
                            mlir::Value val,
                            llvm::SmallVector<mlir::Value> *shapes) {
  /**
   * Get the size of dynamic dims (memref or tensor)
   *
   * Param:
   * mlir::OpBuilder builder, to support MLIR-API calls.
   * mlir::Location loc, to support MLIR-API calls.
   * mlir::Value val, the value to be calculated.
   * llvm::SmallVector<mlir::Value>* shapes, the llvm::SmallVector that stores
   * the sizes of dynamic dimensions.
   *
   * Return:
   * None
   */
  mlir::ShapedType sTy = val.getType().dyn_cast<mlir::ShapedType>();
  llvm::ArrayRef<long int> shape = sTy.getShape();
  for (int64_t i = 0; i < shape.size(); i++) {
    if (shape[i] < 0) {
      // use memref.dim to get the corresponding size
      mlir::Value curDim = ConstantPool::getValue(i, builder.getIndexType());
      mlir::Value curSize;
      if (llvm::isa<mlir::MemRefType>(val.getType()))
        curSize =
            builder.create<mlir::memref::DimOp>(loc, val, curDim).getResult();
      else
        curSize =
            builder.create<mlir::tensor::DimOp>(loc, val, curDim).getResult();
      shapes->push_back(curSize);
      // idx->push_back(i);
    }
  }
  return;
}

void FixUBUtils::getShapes(mlir::OpBuilder builder, mlir::Location loc,
                           mlir::Value val,
                           llvm::SmallVector<mlir::Value> *shapes) {
  /**
   * Get the shape of given mlir::Value. (memref or tensor)
   *
   * Param:
   * mlir::OpBuilder builder, to support MLIR-API calls.
   * mlir::Location loc, to support MLIR-API calls.
   * mlir::Value val, the value to calculate the shape.
   * llvm::SmallVector<mlir::Value>* shapes, the shapes to be returned.
   *
   * Return:
   * None
   */
  mlir::ShapedType sTy = val.getType().dyn_cast<mlir::ShapedType>();
  llvm::ArrayRef<long int> shape = sTy.getShape();
  for (int64_t i = 0; i < shape.size(); i++) {
    mlir::Value curSize;
    if (shape[i] < 0) {
      // use memref.dim to get the corresponding size
      mlir::Value curDim;
      if (llvm::isa<mlir::MemRefType>(val.getType())) {
        curDim = ConstantPool::getValue(i, builder.getIndexType());
        curSize =
            builder.create<mlir::memref::DimOp>(loc, val, curDim).getResult();
      } else {
        curDim = ConstantPool::getValue(i, builder.getIndexType());
        curSize =
            builder.create<mlir::tensor::DimOp>(loc, val, curDim).getResult();
      }

    } else {
      // to get the constant value.
      curSize = ConstantPool::getValue(shape[i], builder.getIndexType());
    }
    shapes->push_back(curSize);
  }
  return;
}

void FixUBUtils::getValidIndex(mlir::OpBuilder builder, mlir::Location loc,
                               llvm::SmallVector<mlir::Value> dims,
                               llvm::SmallVector<mlir::Value> oldIndices,
                               llvm::SmallVector<mlir::Value> *newIndices) {
  /**
   * Get the valid index according to given dims and oldshape.
   * If the oldshape is invalid, new shape will be generated,
   * otherwise, oldshpae will be added into newshape.
   *
   * Params:
   * mlir::OpBuilder builder, to support MLIR API call
   * mlir::Location loc, to support MLIR API call
   * llvm::SmallVector<mlir::Value> dims, the dimensions that the target op
   * hold. llvm::SmallVector<mlir::Value> oldIndices, the old shapes (index
   * type) llvm::SmallVector<mlir::Value>* newIndices the new shpes, which
   * ensures to be valid, i.e., prevent out-of-bounds.
   */
  for (int64_t i = 0; i < dims.size(); ++i) {
    mlir::Value invalidIndex = builder
                                   .create<mlir::index::CmpOp>(
                                       loc, mlir::index::IndexCmpPredicate::UGE,
                                       oldIndices[i], dims[i])
                                   .getOperation()
                                   ->getResult(0);

    auto GreaterBuilder = [&](mlir::OpBuilder &builder,
                              mlir::Location loc) { // do dim % rank
      mlir::Operation *newDimOp =
          builder.create<mlir::index::RemUOp>(loc, oldIndices[i], dims[i]);
      mlir::Value newDim = newDimOp->getResult(0);
      builder.create<mlir::scf::YieldOp>(loc, newDim);
    };

    auto NotGreaterBuilder =
        [&](mlir::OpBuilder &b,
            mlir::Location loc) { // return original dimension directly
          b.create<mlir::scf::YieldOp>(loc, oldIndices[i]);
        };

    mlir::Operation *ifOp =
        builder
            .create<mlir::scf::IfOp>(loc, invalidIndex, GreaterBuilder,
                                     NotGreaterBuilder)
            .getOperation();
    newIndices->push_back(ifOp->getResult(0));
  }
  return;
}

mlir::Operation *
FixUBUtils::compareShape(mlir::OpBuilder builder, mlir::Location loc,
                         llvm::SmallVector<mlir::Value> shape1,
                         llvm::SmallVector<mlir::Value> shape2) {
  /**
   * Compare two shapes specified by two llvm::SmallVector<mlir::Value>
   * parameters.
   *
   * Param:
   * mlir::OpBuilder builder, to support MLIR-API.
   * mlir::Location loc, to support MLIR-API.
   * llvm::SmallVector<mlir::Value> shape1, shape1.
   * llvm::SmallVector<mlir::Value> shape2, shape2.
   *
   * Return:
   * The operation that get the result whether two shapes are same.
   */
  if (shape1.size() != shape2.size()) {
    return ConstantPool::getValue(0, builder.getI1Type()).getDefiningOp();
  }
  mlir::Value shapeSame = ConstantPool::getValue(1, builder.getI1Type());
  for (int64_t i = 0; i < shape1.size(); i++) {
    mlir::Value curSame =
        builder
            .create<mlir::index::CmpOp>(loc, mlir::index::IndexCmpPredicate::EQ,
                                        shape1[i], shape2[i])
            .getResult();
    shapeSame = builder.create<mlir::arith::AndIOp>(loc, curSame, shapeSame)
                    .getResult();
  }
  return shapeSame.getDefiningOp();
}

mlir::Operation *FixUBUtils::castToAllDyn(mlir::OpBuilder builder,
                                          mlir::Location loc, mlir::Value val) {
  /**
   * This function cast the given value to a value that has all
   * dimensions dynamic. (memref or tensor)
   *
   * Params:
   * mlir::OpBuilder builder, to support MLIR-API call.
   * mlir::Location loc, to support MLIR-API all.
   * mlir::Value val, the value to be cast.
   *
   * Return:
   * The pointer of operaiton, which results is the casted value. (memref or
   * tensor)
   */
  mlir::ShapedType sTy = val.getType().dyn_cast<mlir::ShapedType>();
  int64_t rank = sTy.getRank();

  llvm::SmallVector<int64_t> newShape = llvm::SmallVector<int64_t>();
  for (int64_t i = 0; i < rank; i++) {
    newShape.push_back(mlir::ShapedType::kDynamic);
  }
  mlir::Operation *castOp;
  if (llvm::isa<mlir::MemRefType>(val.getType())) {
    mlir::MemRefType newTy =
        mlir::MemRefType::get(newShape, sTy.getElementType());
    castOp = builder.create<mlir::memref::CastOp>(loc, newTy, val);
  } else {
    mlir::RankedTensorType newTy =
        mlir::RankedTensorType::get(newShape, sTy.getElementType());
    castOp = builder.create<mlir::tensor::CastOp>(loc, newTy, val);
  }

  return castOp;
}

mlir::Operation *FixUBUtils::getDifiningOpIncludeArgs(mlir::OpBuilder builder,
                                                      mlir::Location loc,
                                                      mlir::Value val) {
  /**
   * This function is used to find the defining operation of a value.
   * Including the situation that this value is the block argument.
   *
   * Params:
   * mlir::OpBuilder builder, to support MLIR C++ API
   * mlir::Location loc, to support MLIR C++ API
   * mlir::Value val, the value which we need to find the defining op.
   *
   * Return:
   * The pointer of the defining operation of val.
   */
  if (val.getDefiningOp()) {
    return val.getDefiningOp();
  } else {
    mlir::BlockArgument arg = val.dyn_cast<mlir::BlockArgument>();
    assert(arg);
    mlir::Block *block = arg.getOwner();
    mlir::Operation *parentOp = block->getParentOp();
    unsigned idx = 0;
    while (idx < parentOp->getNumOperands()) {
      if (block->getArgument(idx) == val)
        break;
      idx++;
    }
    return FixUBUtils::getDifiningOpIncludeArgs(builder, loc,
                                                parentOp->getOperand(idx));
  }
}