//
// Created by Stan Wang on 2022/12/8.
//

#include "../include/smith/TypedValuePool.h"

using namespace mlir;

void TypedValuePool::addTypeValue(const TypeValue &typedValue, std::string op) {
  if (mlir::dyn_cast<RankedTensorType>(typedValue.type)) {
    auto t = mlir::dyn_cast<RankedTensorType>(typedValue.type);
    if (t.hasStaticShape()) {
      addStaticShapedTensor(typedValue, op);
    } else {
      addDynamicShapedTensor(typedValue, op);
    }
  } else if (mlir::dyn_cast<MemRefType>(typedValue.type)) {
    if (mlir::dyn_cast<MemRefType>(typedValue.type).hasStaticShape()) {
      addStaticShapedMemref(typedValue, op);
    } else {
      addDynamicShapedMemref(typedValue, op);
    }

  } else if (typedValue.type.isIntOrFloat()) {
    addIntOrFloat(typedValue, op);
  } else if (mlir::dyn_cast<IndexType>(typedValue.type)) {
    addIndex(typedValue, op);
  }
}

void TypedValuePool::addIntOrFloat(const TypeValue &element,
                                   std::string op = "") {
  assert(element.type.isIntOrFloat());
  // check type
  intOrFloatPool.push_back(element);
}

void TypedValuePool::addElement(const TypeValue &element, std::string op = "") {
  assert(element.val);
  bool typeCheck = (mlir::dyn_cast<IntegerType>(element.type) ||
                    mlir::dyn_cast<FloatType>(element.type) ||
                    mlir::dyn_cast<IndexType>(element.type));

  debugPrint("calling from " + op);
  assert(typeCheck && "addElement() require Integer or Float or Index");
  if (element.type.isIntOrFloat()) {
    intOrFloatPool.push_back(element);
  } else {
    indexPool.push_back(element);
  }
}

void TypedValuePool::addRankedMemref(const TypeValue &memref,
                                     std::string op = "") {
  auto memTy = mlir::dyn_cast<MemRefType>(memref.val.getType());
  assert(memTy && memTy.hasRank());
  if (memTy.hasStaticShape()) {
    addStaticShapedMemref(memref, op);
  } else {
    addDynamicShapedMemref(memref, op);
  }
}

void TypedValuePool::addStaticShapedMemref(const TypeValue &memref,
                                           std::string op = "") {
  assert(mlir::isa<MemRefType>(memref.val.getType()) &&
         mlir::dyn_cast<MemRefType>(memref.val.getType()).hasStaticShape());
  debugPrint("calling from: " + op);
  staticShapedMemrefPool.push_back(memref);
}

void TypedValuePool::addDynamicShapedMemref(const TypeValue &memref,
                                            std::string op = "") {
  auto memrefTy = mlir::dyn_cast<MemRefType>(memref.type);
  assert(memrefTy.hasRank());
  if (memrefTy.hasStaticShape()) {
    addStaticShapedMemref(memref, op);
  }
  dynamicShapedMemrefPool.push_back(memref);
}

void TypedValuePool::addRankedTensor(const TypeValue &tensor,
                                     std::string op = "") {
  assert(tensor.val);
  auto t = mlir::dyn_cast<RankedTensorType>(tensor.type);
  if (t.hasStaticShape()) {
    addStaticShapedTensor(tensor, op);
  } else {
    addDynamicShapedTensor(tensor, op);
  }
}

void TypedValuePool::addStaticShapedTensor(const TypeValue &tensor,
                                           std::string op = "") {
  debugPrint("calling from: " + op);
  assert(tensor.val);
  auto t = mlir::dyn_cast<RankedTensorType>(tensor.type);
  if (t && t.hasStaticShape()) {
    staticShapedTensorPool.push_back(tensor);
  } else if (conf.debug) {
    std::cout << "MisType inserted in static shaped tensors, op: " << op
              << std::endl;
  }
}

void TypedValuePool::addDynamicShapedTensor(const TypeValue &dynamicTensor,
                                            std::string op = "") {
  debugPrint("calling from: " + op);
  assert(dynamicTensor.val);
  assert(mlir::dyn_cast<RankedTensorType>(dynamicTensor.type) &&
         "Require RankedTensorType here.");
  dynamicShapedTensorPool.push_back(dynamicTensor);
}

void TypedValuePool::addVector(const TypeValue &vector, std::string op = "") {
  assert(vector.val);
  auto ty = mlir::dyn_cast<VectorType>(vector.type);
  if (ty) {
    vectorPool.push_back(vector);
  } else if (conf.debug) {
    std::cout << "MisType inserted in tensors, op: " << op << std::endl;
  }
}

void TypedValuePool::addIndex(const TypeValue &index, std::string op = "") {
  assert(index.val);
  if (mlir::dyn_cast<IndexType>(index.type)) {
    indexPool.push_back(index);
  } else if (conf.debug) {
    std::cout << "MisType inserted in tensors, op: " << op << std::endl;
  }
}

std::vector<TypeValue>
TypedValuePool::getCandidatesFromIntOrFloats(OpBuilder &builder, Location loc,
                                             Type t) {

  if (intOrFloatPool.empty()) {
    generateElement(builder, loc, t);
  }
  return intOrFloatPool;
}

std::vector<TypeValue> TypedValuePool::getCandidatesFromStaticShapedMemref(
    OpBuilder &builder, Location loc, MemRefType t) {
  assert(t.hasStaticShape());
  if (staticShapedMemrefPool.empty()) {
    generateStaticShapedMemref(builder, loc, t);
  }
  return staticShapedMemrefPool;
}

std::vector<TypeValue> TypedValuePool::getCandidatesFromDynamicShapedMemref(
    OpBuilder &builder, Location loc, MemRefType t) {
  if (dynamicShapedMemrefPool.empty()) {
    generateDynamicShapedMemref(builder, loc, t);
  }
  return dynamicShapedMemrefPool;
}

std::vector<TypeValue>
TypedValuePool::getCandidatesFromVector(mlir::OpBuilder &builder,
                                        mlir::Location loc, mlir::Type t) {
  if (vectorPool.empty()) {
    generateVector(builder, loc, mlir::dyn_cast<VectorType>(t));
  }
  return vectorPool;
}

std::vector<TypeValue>
TypedValuePool::getCandidatesFromStaticShapedTensor(OpBuilder &builder,
                                                    Location loc, Type t) {
  if (staticShapedTensorPool.empty()) {
    generateStaticShapedTensor(builder, loc, mlir::dyn_cast<RankedTensorType>(t));
  }
  return staticShapedTensorPool;
}

std::vector<TypeValue>
TypedValuePool::getCandidatesFromIndex(OpBuilder &builder, Location loc) {
  if (indexPool.empty()) {
    indexPool.push_back(
        TypeValue(builder.getIndexType(),
                  builder.create<arith::ConstantIndexOp>(loc, 1)));
  }
  return indexPool;
}

std::vector<TypeValue>
TypedValuePool::searchCandidatesFrom(std::vector<PoolType> poolTypes,
                                     TypeValueFilter filter) {
  std::vector<TypeValue> candidates;
  for (auto type : poolTypes) {
    std::vector<TypeValue> temp;
    switch (type) {
    case StaticShapedTensor: {
      temp = staticShapedTensorPool;
      break;
    }
    case DynamicShapedTensor: {
      temp = dynamicShapedTensorPool;
      break;
    }
    case StaticShapedMemref: {
      temp = staticShapedMemrefPool;
      break;
    }
    case DynamicShapedMemref: {
      temp = dynamicShapedMemrefPool;
      break;
    }
    case IntOrFloat: {
      temp = intOrFloatPool;
      break;
    }
    case Index: {
      temp = indexPool;
      break;
    }
    case Vector: {
      temp = vectorPool;
      break;
    }
    }
    for (auto tval : temp) {
      if (filter(tval)) {
        candidates.push_back(tval);
      }
    }
  }
  return candidates;
}

TypeValue TypedValuePool::generateInteger(OpBuilder &builder, Location loc,
                                          IntegerType type) {
  auto res = builder.create<arith::ConstantIntOp>(loc, UR(2), type);
  auto tVal = TypeValue(type, res);
  addIntOrFloat(tVal, "");
  return tVal;
}

TypeValue TypedValuePool::generateFloat(OpBuilder &builder, Location loc,
                                        FloatType type) {

  auto one = APFloat(type.getFloatSemantics(), 1);
  auto res = builder.create<arith::ConstantFloatOp>(loc, one, type);
  auto tVal = TypeValue(type, res);
  addIntOrFloat(tVal, "");
  return tVal;
}

TypeValue TypedValuePool::generateIndex(mlir::OpBuilder &builder,
                                        mlir::Location loc) {
  auto res = builder.create<arith::ConstantIndexOp>(loc, 0);
  auto tVal = TypeValue(builder.getIndexType(), res);
  addIndex(tVal, "");
  return tVal;
}

Value TypedValuePool::getConstantIndex(OpBuilder &builder, Location loc,
                                       int num) {
  if (constantIndices.find(num) != constantIndices.end()) {
    assert(llvm::detail::isPresent(constantIndices[num]));
    return constantIndices[num];
  }
  auto res = builder.create<arith::ConstantIndexOp>(loc, num);
  constantIndices.insert(std::make_pair(num, res));
  return res;
}

TypeValue TypedValuePool::generateElement(OpBuilder &builder, Location loc,
                                          Type type) {
  auto iTy = mlir::dyn_cast<IntegerType>(type);
  auto fTy = mlir::dyn_cast<FloatType>(type);
  auto idxTy = mlir::dyn_cast<IndexType>(type);
  assert(iTy || fTy || idxTy);
  if (iTy) {
    return generateInteger(builder, loc, iTy);
  } else if (fTy) {
    return generateFloat(builder, loc, fTy);
  } else {
    return generateIndex(builder, loc);
  }
}

TypeValue TypedValuePool::generateRankedTensor(mlir::OpBuilder &builder,
                                               mlir::Location loc,
                                               mlir::RankedTensorType rtTy) {
  assert(rtTy.hasRank());
  if (rtTy.hasStaticShape()) {
    return generateStaticShapedTensor(builder, loc, rtTy);
  } else {
    return generateDynamicShapedTensor(builder, loc, rtTy);
  }
}

TypeValue TypedValuePool::generateStaticShapedTensor(OpBuilder &builder,
                                                     Location loc,
                                                     RankedTensorType type) {
  assert(type.hasStaticShape());
  auto shapedType = mlir::dyn_cast<ShapedType>(type);
  auto t = builder.create<tensor::EmptyOp>(loc, shapedType.getShape(),
                                           shapedType.getElementType());
  auto tVal = TypeValue(type, t);
  addStaticShapedTensor(tVal, "");
  return tVal;
}

// generate an empty d tensor
TypeValue TypedValuePool::generateDynamicShapedTensor(OpBuilder &builder,
                                                      Location loc,
                                                      RankedTensorType type) {
  auto dynDimNum = type.getNumDynamicDims();
  SmallVector<Value> dynDims;
  for (int i = 0; i < dynDimNum; ++i) {
    dynDims.push_back(sampleTypedValueFrom(getCandidatesFromIndex(builder, loc),
                                           "tensor.empty")
                          .val);
  }
  auto t = builder.create<tensor::EmptyOp>(loc, type, ValueRange(dynDims));
  auto tVal = TypeValue(type, t);
  addDynamicShapedTensor(tVal, "");
  return tVal;
}

TypeValue TypedValuePool::generateRankedMemref(mlir::OpBuilder &builder,
                                               mlir::Location loc,
                                               mlir::MemRefType type) {
  if (type.hasStaticShape()) {
    return generateStaticShapedMemref(builder, loc, type);
  } else {
    return generateDynamicShapedMemref(builder, loc, type);
  }
}

TypeValue TypedValuePool::generateStaticShapedMemref(OpBuilder &builder,
                                                     Location loc,
                                                     MemRefType type) {
  auto memref = builder.create<memref::AllocOp>(loc, type);
  auto tVal = TypeValue(type, memref);
  addStaticShapedMemref(tVal, "");
  return tVal;
}

TypeValue TypedValuePool::generateDynamicShapedMemref(OpBuilder &builder,
                                                      Location loc,
                                                      MemRefType type) {
  auto dyNum = type.getNumDynamicDims();
  SmallVector<Value> dyDims;
  auto idxCandidates = getCandidatesFromIndex(builder, loc);
  for (int i = 0; i < dyNum; ++i) {
    dyDims.push_back(sampleTypedValueFrom(idxCandidates, "memref.alloc").val);
  }
  auto memref = builder.create<memref::AllocOp>(loc, type, dyDims);
  auto tval = TypeValue(memref.getType(), memref);
  addRankedMemref(tval);
  return tval;
}

TypeValue TypedValuePool::generateVector(OpBuilder &builder, Location loc,
                                         VectorType type) {
  auto shapeTy = mlir::dyn_cast<ShapedType>(type);
  auto elemTy = shapeTy.getElementType();
  auto candidates =
      searchCandidatesFrom({PoolType::Index, PoolType::IntOrFloat},
                           [&](TypeValue tVal) { return elemTy == tVal.type; });
  if (candidates.empty()) {
    if (mlir::dyn_cast<IndexType>(elemTy)) {
      candidates.push_back(generateIndex(builder, loc));
    } else {
      candidates.push_back(generateElement(builder, loc, elemTy));
    }
  }
  auto elem = sampleTypedValueFrom(candidates, "vector.broadcast");
  auto vec = builder.create<vector::BroadcastOp>(loc, type, elem.val);
  auto tVal = TypeValue(type, vec);
  addVector(tVal, "");
  return tVal;
}

TypeValue TypedValuePool::generateTypedValue(OpBuilder &builder, Location loc,
                                             Type type) {
  if (mlir::isa<IntegerType>(type)) {
    return generateInteger(builder, loc, mlir::dyn_cast<IntegerType>(type));
  } else if (mlir::isa<FloatType>(type)) {
    return generateFloat(builder, loc, mlir::dyn_cast<FloatType>(type));
  } else if (mlir::isa<MemRefType>(type)) {
    auto memTy = mlir::dyn_cast<MemRefType>(type);
    if (memTy.hasStaticShape()) {
      return generateStaticShapedMemref(builder, loc, memTy);
    } else {
      return generateDynamicShapedMemref(builder, loc, memTy);
    }
  } else if (mlir::isa<RankedTensorType>(type)) {
    auto rtTy = mlir::dyn_cast<RankedTensorType>(type);
    if (rtTy.hasStaticShape()) {
      return generateStaticShapedTensor(builder, loc, rtTy);
    } else {
      return generateDynamicShapedTensor(builder, loc, rtTy);
    }
  } else if (mlir::isa<VectorType>(type)) {
    return generateVector(builder, loc, mlir::dyn_cast<VectorType>(type));
  } else if (mlir::isa<IndexType>(type)) {
    return generateIndex(builder, loc);
  } else {
    std::cout << "unsupported type for new value generation\n";
    exit(-1);
  }
}

// TODO deprecate this
std::vector<TypeValue> searchTypedValueFrom(const TypedValuePool pool,
                                            Type type) {
  std::vector<TypeValue> candidates;
  if (mlir::isa<RankedTensorType>(type)) {
    candidates = searchShapedInputFrom(mlir::dyn_cast<ShapedType>(type),
                                       pool.staticShapedTensorPool);
  } else if (mlir::isa<MemRefType>(type)) {
    candidates = searchShapedInputFrom(mlir::dyn_cast<ShapedType>(type),
                                       pool.staticShapedMemrefPool);
  } else if (mlir::isa<FloatType>(type) || mlir::isa<IntegerType>(type)) {
    candidates = searchTypedValueFrom(pool.intOrFloatPool, type);
  } else if (mlir::isa<VectorType>(type)) {
    candidates = searchTypedValueFrom(pool.vectorPool, type);
  } else if (mlir::isa<IndexType>(type)) {
    candidates = searchTypedValueFrom(pool.indexPool, type);
  } else {
    llvm::errs() << "unsupported type" << type << "\n";
    exit(-1);
  }
  return candidates;
}

SmallVector<Value>
TypedValuePool::randomIndicesForShapedType(ShapedType shapedType,
                                           OpBuilder &builder, Location loc) {

  SmallVector<Value, 8> indices;
  for (size_t dim = 0; dim < shapedType.getRank(); ++dim) {
    if (shapedType.isDynamicDim(dim)) {
      indices.push_back(constantIndices.begin()->second);
    } else {
      auto idx = UR(shapedType.getDimSize(dim));
      auto idxVal = getConstantIndex(builder, loc, idx);
      indices.push_back(idxVal);
    }
  }
  return indices;
}