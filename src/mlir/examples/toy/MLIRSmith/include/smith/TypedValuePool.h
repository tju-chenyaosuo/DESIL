//
// Created by Stan Wang on 2022/12/8.
//

#ifndef MLIR_FUZZ_TYPEDVALUEPOOL_H
#define MLIR_FUZZ_TYPEDVALUEPOOL_H

#include "Util.h"

enum PoolType {
  StaticShapedTensor,
  DynamicShapedTensor,
  StaticShapedMemref,
  DynamicShapedMemref,
  IntOrFloat,
  Vector,
  Index
};

inline std::vector<PoolType> allTypes = {PoolType::StaticShapedTensor,
                                         PoolType::DynamicShapedTensor,
                                         PoolType::StaticShapedMemref,
                                         PoolType::DynamicShapedMemref,
                                         PoolType::IntOrFloat,
                                         PoolType::Vector,
                                         PoolType::Index};

using InitValueGenerator =
    std::function<TypeValue(mlir::OpBuilder &, mlir::Location)>;

struct TypedValuePool {
  std::vector<TypeValue> staticShapedTensorPool;
  std::vector<TypeValue> dynamicShapedTensorPool;
  std::vector<TypeValue> staticShapedMemrefPool;
  std::vector<TypeValue> dynamicShapedMemrefPool;
  std::vector<TypeValue> unrankedTensorPool;
  std::vector<TypeValue> unrankedMemrefPool;
  std::vector<TypeValue> intOrFloatPool;
  std::vector<TypeValue> vectorPool;
  std::vector<TypeValue> indexPool;
  std::map<int, mlir::Value> constantIndices;
  // for linalg.generic
  std::vector<llvm::SmallVector<mlir::AffineExpr>> affineExprPool;
  std::vector<mlir::AffineMap> affineMapPool;
  std::vector<mlir::IntegerSet> integerSetPool;

  TypedValuePool(
      std::vector<TypeValue> staticShapedTensorPool = std::vector<TypeValue>(),
      std::vector<TypeValue> dynamicShapedTensorPool = std::vector<TypeValue>(),
      std::vector<TypeValue> staticShapedMemrefPool = std::vector<TypeValue>(),
      std::vector<TypeValue> dynamicShapedMemrefPool = std::vector<TypeValue>(),
      std::vector<TypeValue> unrankedTensorPool = std::vector<TypeValue>(),
      std::vector<TypeValue> unrankedMemrefPool = std::vector<TypeValue>(),
      std::vector<TypeValue> intOrFloatPool = std::vector<TypeValue>(),
      std::vector<TypeValue> vectorPool = std::vector<TypeValue>(),
      std::vector<TypeValue> indexPool = std::vector<TypeValue>(),
      std::map<int, mlir::Value> constantIndices = std::map<int, mlir::Value>(),
      std::vector<llvm::SmallVector<mlir::AffineExpr>> affineExprPool =
          std::vector<llvm::SmallVector<mlir::AffineExpr>>(),
      std::vector<mlir::AffineMap> affineMapPool =
          std::vector<mlir::AffineMap>(),
      std::vector<mlir::IntegerSet> integerSetPool =
          std::vector<mlir::IntegerSet>())
      : staticShapedTensorPool(staticShapedTensorPool),
        dynamicShapedTensorPool(dynamicShapedTensorPool),
        staticShapedMemrefPool(staticShapedMemrefPool),
        dynamicShapedMemrefPool(dynamicShapedMemrefPool),
        unrankedTensorPool(unrankedTensorPool),
        unrankedMemrefPool(unrankedMemrefPool), intOrFloatPool(intOrFloatPool),
        vectorPool(vectorPool), indexPool(indexPool),
        constantIndices(constantIndices), affineExprPool(affineExprPool),
        affineMapPool(affineMapPool), integerSetPool(integerSetPool) {}

  void clearAll() {
    staticShapedTensorPool.clear();
    dynamicShapedTensorPool.clear();
    staticShapedMemrefPool.clear();
    dynamicShapedMemrefPool.clear();
    unrankedMemrefPool.clear();
    unrankedTensorPool.clear();
    intOrFloatPool.clear();
    vectorPool.clear();
    indexPool.clear();
    constantIndices.clear();
    affineExprPool.clear();
    affineMapPool.clear();
    integerSetPool.clear();
  }

  void merge(TypedValuePool &pool) {
    staticShapedTensorPool.insert(staticShapedTensorPool.end(),
                                  pool.staticShapedTensorPool.begin(),
                                  pool.staticShapedTensorPool.end());
    dynamicShapedTensorPool.insert(dynamicShapedTensorPool.end(),
                                   pool.dynamicShapedTensorPool.begin(),
                                   pool.dynamicShapedTensorPool.end());
    staticShapedMemrefPool.insert(staticShapedMemrefPool.end(),
                                  pool.staticShapedMemrefPool.begin(),
                                  pool.staticShapedMemrefPool.end());
    dynamicShapedMemrefPool.insert(dynamicShapedMemrefPool.end(),
                                   pool.dynamicShapedMemrefPool.begin(),
                                   pool.dynamicShapedMemrefPool.end());
    unrankedTensorPool.insert(unrankedTensorPool.begin(),
                              pool.unrankedTensorPool.begin(),
                              pool.unrankedTensorPool.end());
    unrankedMemrefPool.insert(unrankedMemrefPool.begin(),
                              pool.unrankedMemrefPool.begin(),
                              pool.unrankedMemrefPool.end());
    intOrFloatPool.insert(intOrFloatPool.end(), pool.intOrFloatPool.begin(),
                          pool.intOrFloatPool.end());
    indexPool.insert(indexPool.end(), pool.indexPool.begin(),
                     pool.indexPool.end());
    vectorPool.insert(vectorPool.end(), pool.vectorPool.begin(),
                      pool.vectorPool.end());
    constantIndices.insert(pool.constantIndices.begin(),
                           pool.constantIndices.end());
    affineExprPool.insert(affineExprPool.end(), pool.affineExprPool.begin(),
                          pool.affineExprPool.end());
    affineMapPool.insert(affineMapPool.end(), pool.affineMapPool.begin(),
                         pool.affineMapPool.end());
    integerSetPool.insert(integerSetPool.begin(), pool.integerSetPool.begin(),
                          pool.integerSetPool.end());
  }

  void addTypeValue(const TypeValue &typedValue, std::string op);
  void addIntOrFloat(const TypeValue &val, std::string op);
  void addElement(const TypeValue &element, std::string op);
  void addRankedMemref(const TypeValue &memref, std::string op);
  void addUnrankedMemref(const TypeValue &unrankedTensor, std::string op);
  void addStaticShapedMemref(const TypeValue &memref, std::string op);
  void addDynamicShapedMemref(const TypeValue &memref, std::string op);
  void addRankedTensor(const TypeValue &tensor, std::string op);
  void addUnrankedTensor(const TypeValue &unrankedTensor, std::string op);
  void addStaticShapedTensor(const TypeValue &staticTensor, std::string op);
  void addDynamicShapedTensor(const TypeValue &dynamicTensor, std::string op);
  void addVector(const TypeValue &vector, std::string op);
  void addIndex(const TypeValue &index, std::string op);

  /* Giving TypeValue constraints, get candidates from pool  */
  std::vector<TypeValue> getCandidatesFromIntOrFloats(mlir::OpBuilder &builder,
                                                      mlir::Location loc,
                                                      mlir::Type t);
  std::vector<TypeValue>
  getCandidatesFromStaticShapedMemref(mlir::OpBuilder &builder,
                                      mlir::Location loc, mlir::MemRefType t);

  std::vector<TypeValue>
  getCandidatesFromDynamicShapedMemref(mlir::OpBuilder &builder,
                                       mlir::Location loc, mlir::MemRefType t);
  std::vector<TypeValue>
  getCandidatesFromStaticShapedTensor(mlir::OpBuilder &builder,
                                      mlir::Location loc, mlir::Type t);
  std::vector<TypeValue>
  getCandidatesFromDynamicShapedTensor(mlir::OpBuilder &builder,
                                       mlir::Location loc, mlir::Type t);
  std::vector<TypeValue> getCandidatesFromVector(mlir::OpBuilder &builder,
                                                 mlir::Location loc,
                                                 mlir::Type t);
  std::vector<TypeValue> getCandidatesFromIndex(mlir::OpBuilder &builder,
                                                mlir::Location loc);
  std::vector<TypeValue>
  searchCandidatesFrom(std::vector<PoolType>,
                       TypeValueFilter typeValueFilter = emptyFilter());

  /* generate a new value corresponding to provided type, add the return value
   * to the pool */
  TypeValue generateInteger(mlir::OpBuilder &builder, mlir::Location loc,
                            mlir::IntegerType type);

  TypeValue generateFloat(mlir::OpBuilder &builder, mlir::Location loc,
                          mlir::FloatType type);

  TypeValue generateIndex(mlir::OpBuilder &builder, mlir::Location loc);
  mlir::Value getConstantIndex(mlir::OpBuilder &builder, mlir::Location loc,
                               int num);

  TypeValue generateElement(mlir::OpBuilder &builder, mlir::Location loc,
                            mlir::Type type);

  TypeValue generateRankedTensor(mlir::OpBuilder &builder, mlir::Location loc,
                                 mlir::RankedTensorType type);

  // generate an empty s tensor
  TypeValue generateStaticShapedTensor(mlir::OpBuilder &builder,
                                       mlir::Location loc,
                                       mlir::RankedTensorType type);
  // generate an empty d tensor
  TypeValue generateDynamicShapedTensor(mlir::OpBuilder &builder,
                                        mlir::Location loc,
                                        mlir::RankedTensorType type);

  // generate an empty memref, assume it is ranked.
  TypeValue generateRankedMemref(mlir::OpBuilder &builder, mlir::Location loc,
                                 mlir::MemRefType type);
  TypeValue generateStaticShapedMemref(mlir::OpBuilder &builder,
                                       mlir::Location loc,
                                       mlir::MemRefType type);

  TypeValue generateDynamicShapedMemref(mlir::OpBuilder &builder,
                                        mlir::Location loc,
                                        mlir::MemRefType type);

  // generate an empty vector
  TypeValue generateVector(mlir::OpBuilder &builder, mlir::Location loc,
                           mlir::VectorType type);

  TypeValue generateTypedValue(mlir::OpBuilder &builder, mlir::Location loc,
                               mlir::Type type);

  llvm::SmallVector<mlir::Value>
  randomIndicesForShapedType(mlir::ShapedType shapedType,
                             mlir::OpBuilder &builder, mlir::Location loc);
};

TypeValue sampleTypedValueFrom(TypedValuePool pool);

std::vector<TypeValue> searchTypedValueFrom(const TypedValuePool pool,
                                            mlir::Type type);

#endif // MLIR_FUZZ_TYPEDVALUEPOOL_H
