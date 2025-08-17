//
// Created by Stan Wang on 2023/8/17.
//

#include "include/smith/RegionGeneration.h"
#include "include/smith/TypeGeneration.h"
#include "include/smith/generators/OpGeneration.h"

#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"

using namespace mlir;

// Note： T should be either Integer or Float type.
template <typename T>
bool isValidUnaryOperand(TypeValue t, std::vector<uint64_t> validWidths,
                         std::vector<uint64_t> validDims) {
  auto intTy = mlir::dyn_cast<T>(t.type);
  auto vecTy = mlir::dyn_cast<VectorType>(t.type);
  return (intTy && std::find(validWidths.begin(), validWidths.end(),
                             intTy.getWidth()) != validWidths.end()) ||
         (vecTy && mlir::dyn_cast<T>(vecTy.getElementType()) &&
          std::find(validWidths.begin(), validWidths.end(),
                    vecTy.getElementType().getIntOrFloatBitWidth()) !=
              validWidths.end() &&
          vecTy.getRank() == 1 &&
          std::find(validDims.begin(), validDims.end(), vecTy.getDimSize(0)) !=
              validDims.end());
}

bool isBoolOrBoolVector(TypeValue t, std::vector<uint64_t> validDims) {
  auto i = t.type.isInteger(1);
  auto v = mlir::dyn_cast<VectorType>(t.type);
  return i || (v && v.getElementType().isInteger(1) && v.getRank() == 1 &&
               std::find(validDims.begin(), validDims.end(), v.getDimSize(0)) !=
                   validDims.end());
}

bool isIntOrIntVector(TypeValue t, std::vector<uint64_t> validWidths,
                      std::vector<uint64_t> validDims) {
  auto i = mlir::dyn_cast<IntegerType>(t.val.getType());
  auto v = mlir::dyn_cast<VectorType>(t.val.getType());
  return (i && std::find(validWidths.begin(), validWidths.end(),
                         i.getWidth()) != validWidths.end()) ||
         (v && mlir::dyn_cast<IntegerType>(v.getElementType()) && v.getRank() == 1 &&
          std::find(validDims.begin(), validDims.end(), v.getDimSize(0)) !=
              validDims.end());
}

bool isFloatOrFloatVector(TypeValue t, std::vector<uint64_t> validWidths,
                          std::vector<uint64_t> validDims) {
  auto f = mlir::dyn_cast<FloatType>(t.val.getType());
  auto v = mlir::dyn_cast<VectorType>(t.val.getType());
  return (f && std::find(validWidths.begin(), validWidths.end(),
                         f.getWidth()) != validWidths.end()) ||
         (v && mlir::dyn_cast<FloatType>(v.getElementType()) && v.getRank() == 1 &&
          std::find(validDims.begin(), validDims.end(), v.getDimSize(0)) !=
              validDims.end());
}

template <typename T>
OpGenerator getSPIRVIntUnaryOpGenerator(std::string opName) {
  return [&](OpBuilder &builder, Location loc, OpRegion &region) {
    auto candidates = region.pool.searchCandidatesFrom(
        {PoolType::Vector, PoolType::IntOrFloat}, [](TypeValue t) {
          return isValidUnaryOperand<IntegerType>(t, {8, 16, 32, 64},
                                                  {2, 3, 4, 8, 16});
        });
    if (candidates.empty()) {
      candidates.push_back(
          region.pool.generateInteger(builder, loc, builder.getI32Type()));
    }
    auto operand = sampleTypedValueFrom(candidates, opName);
    auto op = builder.create<T>(loc, operand.val);
    auto tval = TypeValue(op.getType(), op);
    region.pool.addTypeValue(tval, opName);
    return op.getOperation();
  };
}

template <typename T>
OpGenerator getSPIRVFloatUnaryOpGenerator(std::string opName) {
  return [&](OpBuilder &builder, Location loc, OpRegion &region) {
    auto candidates = region.pool.searchCandidatesFrom(
        {PoolType::Vector, PoolType::IntOrFloat}, [](TypeValue t) {
          return isValidUnaryOperand<FloatType>(t, {8, 16, 32, 64},
                                                {2, 3, 4, 8, 16});
        });
    if (candidates.empty()) {
      candidates.push_back(
          region.pool.generateFloat(builder, loc, builder.getF32Type()));
    }
    auto operand = sampleTypedValueFrom(candidates, opName);
    auto op = builder.create<T>(loc, operand.val);
    auto tval = TypeValue(op.getType(), op);
    region.pool.addTypeValue(tval, opName);
    return op.getOperation();
  };
}

template <typename T>
OpGenerator getSPIRVFloat16Or32UnaryOpGenerator(std::string opName) {
  return [&](OpBuilder &builder, Location loc, OpRegion &region) {
    auto candidates = region.pool.searchCandidatesFrom(
        {PoolType::Vector, PoolType::IntOrFloat}, [](TypeValue t) {
          return isValidUnaryOperand<FloatType>(t, {16, 32}, {2, 3, 4, 8, 16});
        });
    if (candidates.empty()) {
      candidates.push_back(
          region.pool.generateFloat(builder, loc, builder.getF32Type()));
    }
    auto operand = sampleTypedValueFrom(candidates, opName);
    auto op = builder.create<T>(loc, operand.val);
    auto tval = TypeValue(op.getType(), op);
    region.pool.addTypeValue(tval, opName);
    return op.getOperation();
  };
}

template <typename T>
OpGenerator getSPIRVIntBinaryOpGenerator(std::string opName) {
  return [&](OpBuilder &builder, Location loc, OpRegion &region) {
    auto candidates = region.pool.searchCandidatesFrom(
        {PoolType::Vector, PoolType::IntOrFloat}, [](TypeValue t) {
          return isValidUnaryOperand<IntegerType>(t, {8, 16, 32, 64},
                                                  {2, 3, 4, 8, 16});
        });
    if (candidates.empty()) {
      candidates.push_back(
          region.pool.generateInteger(builder, loc, builder.getI32Type()));
    }
    auto operand = sampleTypedValueFrom(candidates, opName);

    auto operand2Candidates = region.pool.searchCandidatesFrom(
        {PoolType::Vector, PoolType::IntOrFloat},
        [&](TypeValue t) { return t.type == operand.val.getType(); });
    auto op = builder.create<T>(
        loc, operand.val, sampleTypedValueFrom(operand2Candidates, opName).val);
    auto tval = TypeValue(op.getType(), op);
    region.pool.addTypeValue(tval, opName);
    return op.getOperation();
  };
}

template <typename T>
OpGenerator getSPIRVIntLogicalBinaryOpGenerator(std::string opName) {
  return [&](OpBuilder &builder, Location loc, OpRegion &region) {
    auto candidates = region.pool.searchCandidatesFrom(
        {PoolType::Vector, PoolType::IntOrFloat}, [](TypeValue t) {
          return isValidUnaryOperand<IntegerType>(t, {8, 16, 32, 64},
                                                  {2, 3, 4, 8, 16});
        });
    if (candidates.empty()) {
      candidates.push_back(
          region.pool.generateInteger(builder, loc, builder.getI32Type()));
    }
    auto operand = sampleTypedValueFrom(candidates, opName);
    Type resTy = builder.getI1Type();
    auto operandTy = mlir::dyn_cast<VectorType>(operand.val.getType());
    if (operandTy) {
      resTy = VectorType::get(operandTy.getShape(), builder.getI1Type());
    }

    auto operand2Candidates = region.pool.searchCandidatesFrom(
        {PoolType::Vector, PoolType::IntOrFloat},
        [&](TypeValue t) { return t.type == operand.val.getType(); });
    auto op =
        builder.create<T>(loc, resTy, operand.val,
                          sampleTypedValueFrom(operand2Candidates, opName).val);
    auto tval = TypeValue(op.getType(), op);
    region.pool.addTypeValue(tval, opName);
    return op.getOperation();
  };
}
template <typename T>
OpGenerator getSPIRVFloatBinaryOpGenerator(std::string opName) {
  return [&](OpBuilder &builder, Location loc, OpRegion &region) {
    auto candidates = region.pool.searchCandidatesFrom(
        {PoolType::Vector, PoolType::IntOrFloat}, [](TypeValue t) {
          return isValidUnaryOperand<FloatType>(t, {8, 16, 32, 64},
                                                {2, 3, 4, 8, 16});
        });
    if (candidates.empty()) {
      candidates.push_back(
          region.pool.generateFloat(builder, loc, builder.getF32Type()));
    }
    auto operand = sampleTypedValueFrom(candidates, opName);

    auto operand2Candidates = region.pool.searchCandidatesFrom(
        {PoolType::Vector, PoolType::IntOrFloat},
        [&](TypeValue t) { return t.type == operand.val.getType(); });
    auto op = builder.create<T>(
        loc, operand.val, sampleTypedValueFrom(operand2Candidates, opName).val);
    auto tval = TypeValue(op.getType(), op);
    region.pool.addTypeValue(tval, opName);
    return op.getOperation();
  };
}

template <typename T>
OpGenerator getSPIRVBoolUnaryOpGenerator(std::string opName) {
  return [&](OpBuilder &builder, Location loc, OpRegion &region) {
    auto candidates = region.pool.searchCandidatesFrom(
        {PoolType::Vector, PoolType::IntOrFloat},
        [](TypeValue t) { return isBoolOrBoolVector(t, {2, 3, 4, 8, 16}); });
    if (candidates.empty()) {
      candidates.push_back(
          region.pool.generateInteger(builder, loc, builder.getI1Type()));
    }
    auto operand = sampleTypedValueFrom(candidates, opName);
    auto op = builder.create<T>(loc, operand.val);
    auto tval = TypeValue(op.getType(), op);
    region.pool.addTypeValue(tval, opName);
    return op.getOperation();
  };
}

template <typename T>
OpGenerator getSPIRVFUnaryOpGenerator(std::string opName) {
  return [&](OpBuilder &builder, Location loc, OpRegion &region) {
    auto candidates = region.pool.searchCandidatesFrom(
        {PoolType::IntOrFloat, PoolType::Vector}, [](TypeValue t) {
          return isFloatOrFloatVector(t, {16, 32, 64}, {2, 3, 4, 8, 16});
        });
    if (candidates.empty()) {
      candidates.push_back(
          region.pool.generateFloat(builder, loc, builder.getF32Type()));
    }
    auto operand = sampleTypedValueFrom(candidates, opName);
    auto op = builder.create<T>(loc, operand.val);
    auto tval = TypeValue(op.getType(), op);
    region.pool.addTypeValue(tval, opName);
    return op.getOperation();
  };
}

template <typename T>
OpGenerator getSPIRVIUnaryOpGenerator(std::string opName) {
  return [&](OpBuilder &builder, Location loc, OpRegion &region) {
    auto candidates = region.pool.searchCandidatesFrom(
        {PoolType::IntOrFloat, PoolType::Vector}, [](TypeValue t) {
          return isIntOrIntVector(t, {8, 16, 32, 64}, {2, 3, 4, 8, 16});
        });
    if (candidates.empty()) {
      candidates.push_back(
          region.pool.generateInteger(builder, loc, builder.getI32Type()));
    }
    auto operand = sampleTypedValueFrom(candidates, opName);
    auto op = builder.create<T>(loc, operand.val);
    auto tval = TypeValue(op.getType(), op);
    region.pool.addTypeValue(tval, opName);
    return op.getOperation();
  };
}

OpGenerator spirvBitCountGenerator() {
  return getSPIRVIntUnaryOpGenerator<spirv::BitCountOp>("spirv.BitCount");
}

OpGenerator spirvBitFieldInsertGenerator() {
  return [&](OpBuilder &builder, Location loc, OpRegion &region) {
    auto opName = "spirv.BitFieldInsert";
    auto vecCandidates =
        region.pool.searchCandidatesFrom({PoolType::Vector}, [](TypeValue t) {
          if (mlir::dyn_cast<VectorType>(t.val.getType()).getElementType().isInteger(
                  1)) {
            return false;
          }
          return isIntOrIntVector(t, {8, 16, 32, 64}, {2, 3, 4, 8, 16});
        });
    if (vecCandidates.empty()) {
      vecCandidates.push_back(region.pool.generateVector(
          builder, loc, VectorType::get({2}, builder.getI32Type())));
    }
    auto operand1 = sampleTypedValueFrom(vecCandidates, opName);
    auto operand2Candidates =
        region.pool.searchCandidatesFrom({PoolType::Vector}, [&](TypeValue t) {
          return t.type == operand1.val.getType();
        });
    auto operand2 = sampleTypedValueFrom(operand2Candidates, opName);

    auto intCandidates = region.pool.searchCandidatesFrom(
        {PoolType::IntOrFloat}, [](TypeValue t) {
          auto ty = mlir::dyn_cast<IntegerType>(t.val.getType());
          return ty && (ty.getWidth() == 8 || ty.getWidth() == 16 ||
                        ty.getWidth() == 32 || ty.getWidth() == 64);
        });
    if (intCandidates.empty()) {
      intCandidates.push_back(
          region.pool.generateInteger(builder, loc, builder.getI32Type()));
    }
    auto operand3 = sampleTypedValueFrom(intCandidates, opName);
    auto operand4 = sampleTypedValueFrom(intCandidates, opName);

    auto op = builder.create<spirv::BitFieldInsertOp>(
        loc, operand1.val, operand2.val, operand3.val, operand4.val);
    auto tval = TypeValue(op.getType(), op);
    region.pool.addTypeValue(tval, opName);
    return op.getOperation();
  };
}

template <typename T>
OpGenerator spirvBitFieldExtractGenerator(std::string uOrS) {
  return [&](OpBuilder &builder, Location loc, OpRegion &region) {
    auto opName = "spirv.BitField" + uOrS + "Extract";
    auto vecCandidates =
        region.pool.searchCandidatesFrom({PoolType::Vector}, [](TypeValue t) {
          if (mlir::dyn_cast<VectorType>(t.val.getType()).getElementType().isInteger(
                  1)) {
            return false;
          }
          return isIntOrIntVector(t, {8, 16, 32, 64}, {2, 3, 4, 8, 16});
        });
    if (vecCandidates.empty()) {
      vecCandidates.push_back(region.pool.generateVector(
          builder, loc, VectorType::get({2}, builder.getI32Type())));
    }
    auto operand1 = sampleTypedValueFrom(vecCandidates, opName);

    auto intCandidates = region.pool.searchCandidatesFrom(
        {PoolType::IntOrFloat}, [](TypeValue t) {
          auto ty = mlir::dyn_cast<IntegerType>(t.val.getType());
          return ty && (ty.getWidth() == 8 || ty.getWidth() == 16 ||
                        ty.getWidth() == 32 || ty.getWidth() == 64);
        });
    if (intCandidates.empty()) {
      intCandidates.push_back(
          region.pool.generateInteger(builder, loc, builder.getI32Type()));
    }
    auto operand3 = sampleTypedValueFrom(intCandidates, opName);
    auto operand4 = sampleTypedValueFrom(intCandidates, opName);

    auto op = builder.create<T>(loc, operand1.val, operand3.val, operand4.val);
    auto tval = TypeValue(op.getType(), op);
    region.pool.addTypeValue(tval, opName);
    return op.getOperation();
  };
}

OpGenerator spirvBitFieldSExtractGenerator() {
  return spirvBitFieldExtractGenerator<spirv::BitFieldSExtractOp>("S");
}

OpGenerator spirvBitFieldUExtractGenerator() {
  return spirvBitFieldExtractGenerator<spirv::BitFieldUExtractOp>("U");
}

OpGenerator spirvBitReverseGenerator() {
  return getSPIRVIntUnaryOpGenerator<spirv::BitReverseOp>("spirv.BitReverse");
}

OpGenerator spirvNotGenerator() {
  return getSPIRVIntUnaryOpGenerator<spirv::NotOp>("spirv.Not");
}

template <typename T>
OpGenerator getSPIRVBitBinaryOpGenerator(std::string opName) {
  return [&](OpBuilder &builder, Location loc, OpRegion &region) {
    auto vecCandidates =
        region.pool.searchCandidatesFrom({PoolType::Vector}, [](TypeValue t) {
          if (mlir::dyn_cast<VectorType>(t.val.getType()).getElementType().isInteger(
                  1)) {
            return false;
          }
          return isIntOrIntVector(t, {8, 16, 32, 64}, {2, 3, 4, 8, 16});
        });
    if (vecCandidates.empty()) {
      vecCandidates.push_back(region.pool.generateVector(
          builder, loc, VectorType::get({2}, builder.getI32Type())));
    }
    auto operand1 = sampleTypedValueFrom(vecCandidates, opName);

    auto operand2Candidates =
        region.pool.searchCandidatesFrom({PoolType::Vector}, [&](TypeValue t) {
          return t.type == operand1.val.getType();
        });
    auto operand2 = sampleTypedValueFrom(operand2Candidates, opName);

    auto op = builder.create<T>(loc, operand1.val, operand2.val);
    auto tval = TypeValue(op.getType(), op);
    region.pool.addTypeValue(tval, opName);
    return op.getOperation();
  };
}

OpGenerator spirvBitwiseAndGenerator() {
  return getSPIRVBitBinaryOpGenerator<spirv::BitwiseAndOp>("spirv.BitwiseAnd");
}

OpGenerator spirvBitwiseOrGenerator() {
  return getSPIRVBitBinaryOpGenerator<spirv::BitwiseOrOp>("spirv.BitwiseOr");
}

OpGenerator spirvBitwiseXorGenerator() {
  return getSPIRVBitBinaryOpGenerator<spirv::BitwiseXorOp>("spirv.BitwiseXor");
}

OpGenerator spirvCLCeilGenerator() {
  return getSPIRVFloatUnaryOpGenerator<spirv::CLCeilOp>("spirv.CL.ceil");
}

OpGenerator spirvCLCosGenerator() {
  return getSPIRVFloatUnaryOpGenerator<spirv::CLCosOp>("spirv.CL.cos");
}

OpGenerator spirvCLErfGenerator() {
  return getSPIRVFloatUnaryOpGenerator<spirv::CLErfOp>("spirv.CL.erf");
}

OpGenerator spirvCLExpGenerator() {
  return getSPIRVFloatUnaryOpGenerator<spirv::CLExpOp>("spirv.CL.exp");
}

OpGenerator spirvCLFAbsGenerator() {
  return getSPIRVFloatUnaryOpGenerator<spirv::CLFAbsOp>("spirv.CL.fabs");
}

OpGenerator spirvCLFloorGenerator() {
  return getSPIRVFloatUnaryOpGenerator<spirv::CLFloorOp>("spirv.CL.floor");
}

OpGenerator spirvCLLogGenerator() {
  return getSPIRVFloatUnaryOpGenerator<spirv::CLLogOp>("spirv.CL.log");
}

OpGenerator spirvCLRoundGenerator() {
  return getSPIRVFloatUnaryOpGenerator<spirv::CLRoundOp>("spirv.CL.round");
}

OpGenerator spirvCLRintGenerator() {
  return getSPIRVFloatUnaryOpGenerator<spirv::CLRintOp>("spirv.CL.rint");
}

OpGenerator spirvCLRsqrtGenerator() {
  return getSPIRVFloatUnaryOpGenerator<spirv::CLRsqrtOp>("spirv.CL.rsqrt");
}

OpGenerator spirvCLSinGenerator() {
  return getSPIRVFloatUnaryOpGenerator<spirv::CLSinOp>("spirv.CL.sin");
}

OpGenerator spirvCLSqrtGenerator() {
  return getSPIRVFloatUnaryOpGenerator<spirv::CLSqrtOp>("spirv.CL.sqrt");
}

OpGenerator spirvCLTanhGenerator() {
  return getSPIRVFloatUnaryOpGenerator<spirv::CLTanhOp>("spirv.CL.tanh");
}

OpGenerator spirvCLSAbsGenerator() {
  return getSPIRVIntUnaryOpGenerator<spirv::CLSAbsOp>("spirv.CL.sabs");
}

OpGenerator spirvCLFMaxGenerator() {
  return getSPIRVFloatBinaryOpGenerator<spirv::CLFMaxOp>("spirv.CL.fmax");
}

OpGenerator spirvCLFMinGenerator() {
  return getSPIRVFloatBinaryOpGenerator<spirv::CLFMinOp>("spirv.CL.fmin");
}
OpGenerator spirvCLPowGenerator() {
  return getSPIRVFloatBinaryOpGenerator<spirv::CLPowOp>("spirv.CL.pow");
}

OpGenerator spirvCLSMaxGenerator() {
  return getSPIRVIntBinaryOpGenerator<spirv::CLSMaxOp>("spirv.CL.smax");
}

OpGenerator spirvCLSMinGenerator() {
  return getSPIRVIntBinaryOpGenerator<spirv::CLSMinOp>("spirv.CL.smin");
}

OpGenerator spirvCLUMaxGenerator() {
  return getSPIRVIntBinaryOpGenerator<spirv::CLUMaxOp>("spirv.CL.umax");
}

OpGenerator spirvCLUMinGenerator() {
  return getSPIRVIntBinaryOpGenerator<spirv::CLUMinOp>("spirv.CL.umin");
}

template <typename T>
OpGenerator getSPIRVFloatTriOpGenerator(std::string opName) {
  return [&](OpBuilder &builder, Location loc, OpRegion &region) {
    auto candidates = region.pool.searchCandidatesFrom(
        {PoolType::Vector, PoolType::IntOrFloat}, [](TypeValue t) {
          return isValidUnaryOperand<FloatType>(t, {16, 32, 64},
                                                {2, 3, 4, 8, 16});
        });
    if (candidates.empty()) {
      candidates.push_back(
          region.pool.generateFloat(builder, loc, builder.getF32Type()));
    }
    auto operand = sampleTypedValueFrom(candidates, opName);

    auto operand2Candidates = region.pool.searchCandidatesFrom(
        {PoolType::Vector, PoolType::IntOrFloat},
        [&](TypeValue t) { return t.type == operand.val.getType(); });
    auto op =
        builder.create<T>(loc, operand.val.getType(), operand.val,
                          sampleTypedValueFrom(operand2Candidates, opName).val,
                          sampleTypedValueFrom(operand2Candidates, opName).val);
    auto tval = TypeValue(op.getType(), op);
    region.pool.addTypeValue(tval, opName);
    return op.getOperation();
  };
}

template <typename T>
OpGenerator getSPIRVIntTriOpGenerator(std::string opName) {
  return [&](OpBuilder &builder, Location loc, OpRegion &region) {
    auto candidates = region.pool.searchCandidatesFrom(
        {PoolType::Vector, PoolType::IntOrFloat}, [](TypeValue t) {
          return isValidUnaryOperand<IntegerType>(t, {8, 16, 32, 64},
                                                  {2, 3, 4, 8, 16});
        });
    if (candidates.empty()) {
      candidates.push_back(
          region.pool.generateFloat(builder, loc, builder.getF32Type()));
    }
    auto operand = sampleTypedValueFrom(candidates, opName);

    auto operand2Candidates = region.pool.searchCandidatesFrom(
        {PoolType::Vector, PoolType::IntOrFloat},
        [&](TypeValue t) { return t.type == operand.val.getType(); });
    auto op =
        builder.create<T>(loc, operand.val.getType(), operand.val,
                          sampleTypedValueFrom(operand2Candidates, opName).val,
                          sampleTypedValueFrom(operand2Candidates, opName).val);
    auto tval = TypeValue(op.getType(), op);
    region.pool.addTypeValue(tval, opName);
    return op.getOperation();
  };
}

OpGenerator spirvCLFmaGenerator() {
  return getSPIRVFloatTriOpGenerator<spirv::CLFmaOp>("spriv.CL.fma");
}
// OpGenerator spirvCLPrintfGenerator(){};

// OpGenerator spirvBitcastGenerator(){};
// OpGenerator spirvBranchConditionalGenerator(){};
// OpGenerator spirvBranchGenerator(){};

// OpGenerator spirvCompositeConstructGenerator(){};
// OpGenerator spirvCompositeExtractGenerator(){};
// OpGenerator spirvCompositeInsertGenerator(){};
// OpGenerator spirvConstantGenerator(){};
// OpGenerator spirvControlBarrierGenerator(){};
// OpGenerator spirvConvertFToSGenerator(){};
// OpGenerator spirvConvertFToUGenerator(){};
// OpGenerator spirvConvertPtrToUGenerator(){};
// OpGenerator spirvConvertSToFGenerator(){};
// OpGenerator spirvConvertUToFGenerator(){};
// OpGenerator spirvConvertUToPtrGenerator(){};
// OpGenerator spirvCopyMemoryGenerator(){};
// OpGenerator spirvEXTAtomicFAddGenerator(){};
// OpGenerator spirvEntryPointGenerator(){};
// OpGenerator spirvExecutionModeGenerator(){};
// OpGenerator spirvFAddGenerator(){};
// OpGenerator spirvFConvertGenerator(){};
// OpGenerator spirvFDivGenerator(){};
// OpGenerator spirvFModGenerator(){};
// OpGenerator spirvFMulGenerator(){};

// TODO- matrix
OpGenerator spirvFNegateGenerator() {
  return getSPIRVFUnaryOpGenerator<spirv::FNegateOp>("spirv.FNegate");
}

OpGenerator spirvFOrdEqualGenerator() {
  return getSPIRVFloatBinaryOpGenerator<spirv::FOrdEqualOp>("spirv.FOrdEqual");
}

OpGenerator spirvFOrdGreaterThanEqualGenerator() {
  return getSPIRVFloatBinaryOpGenerator<spirv::FOrdGreaterThanEqualOp>(
      "spriv.FOrdGreaterThanEqual");
}

OpGenerator spirvFOrdGreaterThanGenerator() {
  return getSPIRVFloatBinaryOpGenerator<spirv::FOrdGreaterThanOp>(
      "spriv.FOrdGreaterThan");
}

OpGenerator spirvFOrdLessThanEqualGenerator() {
  return getSPIRVFloatBinaryOpGenerator<spirv::FOrdLessThanEqualOp>(
      "spirv.FOrdLessThanEqual");
}

OpGenerator spirvFOrdLessThanGenerator() {
  return getSPIRVFloatBinaryOpGenerator<spirv::FOrdLessThanOp>(
      "spirv.FOrdLessThan");
}
OpGenerator spirvFOrdNotEqualGenerator() {
  return getSPIRVFloatBinaryOpGenerator<spirv::FOrdNotEqualOp>(
      "spirv.FOrdNotEqual");
}

OpGenerator spirvFUnordEqualGenerator() {
  return getSPIRVFloatBinaryOpGenerator<spirv::FUnordEqualOp>(
      "spirv.FUnordEqual");
}

OpGenerator spirvFUnordGreaterThanEqualGenerator() {
  return getSPIRVFloatBinaryOpGenerator<spirv::FUnordGreaterThanEqualOp>(
      "spirv.FUnordGreaterThanEqual");
}

OpGenerator spirvFUnordGreaterThanGenerator() {
  return getSPIRVFloatBinaryOpGenerator<spirv::FUnordGreaterThanOp>(
      "spirv.FUnordGreaterThan");
}

OpGenerator spirvFUnordLessThanEqualGenerator() {
  return getSPIRVFloatBinaryOpGenerator<spirv::FUnordLessThanEqualOp>(
      "spirv.FUnordLessThanEqual");
}

OpGenerator spirvFUnordLessThanGenerator() {
  return getSPIRVFloatBinaryOpGenerator<spirv::FUnordLessThanOp>(
      "spirv.FUnordLessThan");
}

OpGenerator spirvFUnordNotEqualGenerator() {
  return getSPIRVFloatBinaryOpGenerator<spirv::FUnordNotEqualOp>(
      "spirv.FUnordNotEqual");
}

OpGenerator spirvIEqualGenerator() {
  return getSPIRVIntLogicalBinaryOpGenerator<spirv::IEqualOp>("spirv.IEqual");
}

OpGenerator spirvINotEqualGenerator() {
  return getSPIRVIntLogicalBinaryOpGenerator<spirv::INotEqualOp>(
      "spirv.INotEqual");
}

template <typename T>
OpGenerator getSPIRVBoolBinaryOpGenerator(std::string opName) {
  return [&](OpBuilder &builder, Location loc, OpRegion &region) {
    auto candidates = region.pool.searchCandidatesFrom(
        {PoolType::Vector, PoolType::IntOrFloat}, [](TypeValue t) {
          return isValidUnaryOperand<IntegerType>(t, {1}, {2, 3, 4, 8, 16});
        });
    if (candidates.empty()) {
      candidates.push_back(
          region.pool.generateInteger(builder, loc, builder.getI1Type()));
    }
    auto operand = sampleTypedValueFrom(candidates, opName);
    auto operand2Candidates = region.pool.searchCandidatesFrom(
        {PoolType::Vector, PoolType::IntOrFloat},
        [&](TypeValue t) { return t.type == operand.val.getType(); });
    auto op = builder.create<T>(
        loc, operand.val, sampleTypedValueFrom(operand2Candidates, opName).val);
    auto tval = TypeValue(op.getType(), op);
    region.pool.addTypeValue(tval, opName);
    return op.getOperation();
  };
}

OpGenerator spirvLogicalEqualGenerator() {
  return getSPIRVBoolBinaryOpGenerator<spirv::LogicalEqualOp>(
      "spirv.LogicalEqual");
}

OpGenerator spirvLogicalNotGenerator() {
  return getSPIRVBoolUnaryOpGenerator<spirv::LogicalNotOp>("spirv.LogicalNot");
}

OpGenerator spirvLogicalNotEqualGenerator() {
  return getSPIRVBoolBinaryOpGenerator<spirv::LogicalNotEqualOp>(
      "spirv.LogicalNotEqual");
}

OpGenerator spirvLogicalAndGenerator() {
  return getSPIRVBoolBinaryOpGenerator<spirv::LogicalAndOp>("spirv.LogicalAnd");
}

OpGenerator spirvLogicalOrGenerator() {
  return getSPIRVBoolBinaryOpGenerator<spirv::LogicalOrOp>("spirv.LogicalOr");
}

OpGenerator spirvSGreaterThanEqualGenerator() {
  return getSPIRVIntLogicalBinaryOpGenerator<spirv::SGreaterThanEqualOp>(
      "spirv.SGreaterThanEqual");
}

OpGenerator spirvSGreaterThanGenerator() {
  return getSPIRVIntLogicalBinaryOpGenerator<spirv::SGreaterThanOp>(
      "spirv.SGreaterThan");
}

OpGenerator spirvSLessThanEqualGenerator() {
  return getSPIRVIntLogicalBinaryOpGenerator<spirv::SLessThanEqualOp>(
      "spirv.SLessThanEqual");
}

OpGenerator spirvSLessThanGenerator() {
  return getSPIRVIntLogicalBinaryOpGenerator<spirv::SLessThanOp>(
      "spirv.SLessEqual");
}

OpGenerator spirvUGreaterThanEqualGenerator() {
  return getSPIRVIntLogicalBinaryOpGenerator<spirv::UGreaterThanEqualOp>(
      "spirv.UGreaterThanEqual");
}

OpGenerator spirvUGreaterThanGenerator() {
  return getSPIRVIntLogicalBinaryOpGenerator<spirv::UGreaterThanOp>(
      "spirv.UGreaterThan");
}

OpGenerator spirvULessThanEqualGenerator() {
  return getSPIRVIntLogicalBinaryOpGenerator<spirv::ULessThanEqualOp>(
      "spirv.ULessThanEqual");
}

OpGenerator spirvULessThanGenerator() {
  return getSPIRVIntLogicalBinaryOpGenerator<spirv::ULessThanOp>(
      "spirv.ULessThan");
}

OpGenerator spirvUnorderedGenerator() {
  return getSPIRVFloatBinaryOpGenerator<spirv::UnorderedOp>("spirv.Unordered");
}

// OpGenerator spirvFRemGenerator(){};
// OpGenerator spirvFSubGenerator(){};
// OpGenerator spirvFuncGenerator(){};
// OpGenerator spirvFunctionCallGenerator(){};
OpGenerator spirvGLAcosGenerator() {
  return getSPIRVFloat16Or32UnaryOpGenerator<spirv::GLAcosOp>("spirv.GL.Acos");
}

OpGenerator spirvGLAsinGenerator() {
  return getSPIRVFloat16Or32UnaryOpGenerator<spirv::GLAsinOp>("spirv.GL.Asin");
}

OpGenerator spirvGLAtanGenerator() {
  return getSPIRVFloat16Or32UnaryOpGenerator<spirv::GLAtanOp>("spirv.GL.Atan");
}

OpGenerator spirvGLCeilGenerator() {
  return getSPIRVFloatUnaryOpGenerator<spirv::GLCeilOp>("spirv.GL.Ceil");
}

OpGenerator spirvGLCosGenerator() {
  return getSPIRVFloat16Or32UnaryOpGenerator<spirv::GLCosOp>("spirv.GL.Cos");
}

OpGenerator spirvGLCoshGenerator() {
  return getSPIRVFloat16Or32UnaryOpGenerator<spirv::GLCoshOp>("spirv.GL.Cosh");
}

OpGenerator spirvGLExpGenerator() {
  return getSPIRVFloat16Or32UnaryOpGenerator<spirv::GLExpOp>("spirv.GL.Exp");
}

OpGenerator spirvGLFAbsGenerator() {
  return getSPIRVFloatUnaryOpGenerator<spirv::GLFAbsOp>("spirv.GL.FAbs");
}

OpGenerator spirvGLFSignGenerator() {
  return getSPIRVFloatUnaryOpGenerator<spirv::GLFSignOp>("spirv.GL.FSign");
}

OpGenerator spirvGLFloorGenerator() {
  return getSPIRVFloatUnaryOpGenerator<spirv::GLFloorOp>("spirv.GL.Floor");
}

OpGenerator spirvGLInverseSqrtGenerator() {
  return getSPIRVFloatUnaryOpGenerator<spirv::GLInverseSqrtOp>(
      "spirv.GL.InverseSqrt");
}

OpGenerator spirvGLLogGenerator() {
  return getSPIRVFloat16Or32UnaryOpGenerator<spirv::GLLogOp>("spirv.GL.Log");
}

OpGenerator spirvGLRoundEvenGenerator() {
  return getSPIRVFloatUnaryOpGenerator<spirv::GLRoundEvenOp>(
      "spirv.GL.RoundEven");
}

OpGenerator spirvGLRoundGenerator() {
  return getSPIRVFloatUnaryOpGenerator<spirv::GLRoundOp>("spirv.GL.Round");
}

OpGenerator spirvGLSinGenerator() {
  return getSPIRVFloat16Or32UnaryOpGenerator<spirv::GLSinOp>("spirv.GL.Sin");
}

OpGenerator spirvGLSinhGenerator() {
  return getSPIRVFloat16Or32UnaryOpGenerator<spirv::GLSinhOp>("spirv.GL.Sinh");
}

OpGenerator spirvGLSqrtGenerator() {
  return getSPIRVFloatUnaryOpGenerator<spirv::GLSqrtOp>("spirv.GL.Sqrt");
}

OpGenerator spirvGLTanGenerator() {
  return getSPIRVFloat16Or32UnaryOpGenerator<spirv::GLTanOp>("spirv.GL.Tan");
}

OpGenerator spirvGLTanhGenerator() {
  return getSPIRVFloat16Or32UnaryOpGenerator<spirv::GLTanhOp>("spirv.GL.Tanh");
}

OpGenerator spirvGLFClampGenerator() {
  return getSPIRVFloatTriOpGenerator<spirv::GLFClampOp>("spirv.GL.FClamp");
}

OpGenerator spirvGLSAbsGenerator() {
  return getSPIRVIntUnaryOpGenerator<spirv::GLSAbsOp>("spirv.GL.SAbs");
}

OpGenerator spirvGLSSignGenerator() {
  return getSPIRVIntUnaryOpGenerator<spirv::GLSSignOp>("spirv.GL.SSign");
}

OpGenerator spirvGLFMaxGenerator() {
  return getSPIRVFloatBinaryOpGenerator<spirv::GLFMaxOp>("spirv.GL.FMax");
}

OpGenerator spirvGLFMinGenerator() {
  return getSPIRVFloatBinaryOpGenerator<spirv::GLFMinOp>("spirv.GL.FMin");
}

OpGenerator spirvGLFMixGenerator() {
  return getSPIRVFloatTriOpGenerator<spirv::GLFMixOp>("spirv.GL.FMix");
}

OpGenerator spirvGLFindUMsbGenerator() {
  auto opName = "spirv.GL.FindUMsb";
  return [&](OpBuilder &builder, Location loc, OpRegion &region) {
    auto candidates = region.pool.searchCandidatesFrom(
        {PoolType::Vector, PoolType::IntOrFloat}, [](TypeValue t) {
          return isValidUnaryOperand<IntegerType>(t, {32}, {2, 3, 4, 8, 16});
        });
    if (candidates.empty()) {
      candidates.push_back(
          region.pool.generateInteger(builder, loc, builder.getI32Type()));
    }
    auto operand = sampleTypedValueFrom(candidates, opName);
    auto op = builder.create<spirv::GLFindUMsbOp>(loc, operand.val);
    auto tval = TypeValue(op.getType(), op);
    region.pool.addTypeValue(tval, opName);
    return op.getOperation();
  };
}

OpGenerator spirvGLFmaGenerator() {
  return getSPIRVFloatTriOpGenerator<spirv::GLFmaOp>("spriv.GL.fma");
}

OpGenerator spirvGLLdexpGenerator() {
  auto opName = "spirv.GL.Ldexp";
  return [&](OpBuilder &builder, Location loc, OpRegion &region) {
    auto candidates1 = region.pool.searchCandidatesFrom(
        {PoolType::Vector, PoolType::IntOrFloat}, [](TypeValue t) {
          return isValidUnaryOperand<FloatType>(t, {16, 32, 64},
                                                {2, 3, 4, 8, 16});
        });
    if (candidates1.empty()) {
      candidates1.push_back(
          region.pool.generateFloat(builder, loc, builder.getF32Type()));
    }
    auto operand1 = sampleTypedValueFrom(candidates1, opName);
    auto isVec = false;
    VectorType ty;
    if (mlir::dyn_cast<VectorType>(operand1.val.getType())) {
      isVec = true;
      ty = VectorType::get(
          mlir::dyn_cast<VectorType>(operand1.val.getType()).getShape(),
          builder.getI32Type());
    }
    auto candidates2 = region.pool.searchCandidatesFrom(
        {PoolType::Vector, PoolType::IntOrFloat}, [&](TypeValue t) {
          auto validBitWidth = {8, 16, 32, 64};
          auto i = mlir::dyn_cast<IntegerType>(t.type);
          auto v = mlir::dyn_cast<VectorType>(t.type);
          return (!isVec && i &&
                  std::find(validBitWidth.begin(), validBitWidth.end(),
                            i.getWidth()) != validBitWidth.end()) ||
                 (isVec && v && v.getRank() == 1 &&
                  v.getDimSize(0) ==
                      mlir::dyn_cast<VectorType>(operand1.val.getType()).getDimSize(
                          0) &&
                  mlir::dyn_cast<IntegerType>(v.getElementType()) &&
                  std::find(
                      validBitWidth.begin(), validBitWidth.end(),
                      mlir::dyn_cast<IntegerType>(v.getElementType()).getWidth()) !=
                      validBitWidth.end());
        });
    if (candidates2.empty()) {
      if (isVec) {

        candidates2.push_back(region.pool.generateVector(builder, loc, ty));

      } else {
        candidates2.push_back(
            region.pool.generateInteger(builder, loc, builder.getI32Type()));
      }
    }
    auto operand2 = sampleTypedValueFrom(candidates2, opName);
    auto op = builder.create<spirv::GLLdexpOp>(loc, operand1.val, operand2.val);
    auto tval = TypeValue(op.getType(), op);
    region.pool.addTypeValue(tval, opName);
    return op.getOperation();
  };
}
// TODO-
OpGenerator spirvGLPowGenerator() {
  return [&](OpBuilder &builder, Location loc, OpRegion &region) {
    auto opName = "spirv.GL.Pow";
    auto candidates = region.pool.searchCandidatesFrom(
        {PoolType::Vector, PoolType::IntOrFloat}, [](TypeValue t) {
          return isValidUnaryOperand<FloatType>(t, {16, 32}, {2, 3, 4, 8, 16});
        });
    if (candidates.empty()) {
      candidates.push_back(
          region.pool.generateFloat(builder, loc, builder.getF32Type()));
    }
    auto operand = sampleTypedValueFrom(candidates, opName);

    auto operand2Candidates = region.pool.searchCandidatesFrom(
        {PoolType::Vector, PoolType::IntOrFloat},
        [&](TypeValue t) { return t.type == operand.val.getType(); });
    auto op = builder.create<spirv::GLPowOp>(
        loc, operand.val, sampleTypedValueFrom(operand2Candidates, opName).val);
    auto tval = TypeValue(op.getType(), op);
    region.pool.addTypeValue(tval, opName);
    return op.getOperation();
  };
}

OpGenerator spirvGLSClampGenerator() {
  return getSPIRVIntTriOpGenerator<spirv::GLSClampOp>("spirv.GL.SClamp");
}

OpGenerator spirvGLSMaxGenerator() {
  return getSPIRVIntBinaryOpGenerator<spirv::GLSMaxOp>("spirv.GL.SMax");
}

OpGenerator spirvGLSMinGenerator() {
  return getSPIRVIntBinaryOpGenerator<spirv::GLSMinOp>("spirv.GL.SMin");
}

OpGenerator spirvGLUMaxGenerator() {
  return getSPIRVIntBinaryOpGenerator<spirv::GLUMaxOp>("spirv.GL.UMax");
}

OpGenerator spirvGLUMinGenerator() {
  return getSPIRVIntBinaryOpGenerator<spirv::GLUMaxOp>("spirv.GL.UMin");
}

OpGenerator spirvGLUClampGenerator() {
  return getSPIRVIntTriOpGenerator<spirv::GLUClampOp>("spirv.GL.UClamp");
}
//  OpGenerator spirvGenericCastToPtrExplicitGenerator(){};
//  OpGenerator spirvGenericCastToPtrGenerator(){};
//  OpGenerator spirvGlobalVariableGenerator(){};
//  OpGenerator spirvGroupBroadcastGenerator(){};
//  OpGenerator spirvGroupFAddGenerator(){};
//  OpGenerator spirvGroupFMaxGenerator(){};
//  OpGenerator spirvGroupFMinGenerator(){};
//  OpGenerator spirvGroupFMulKHRGenerator(){};
//  OpGenerator spirvGroupIAddGenerator(){};
//  OpGenerator spirvGroupIMulKHRGenerator(){};
//  OpGenerator spirvGroupNonUniformBallotGenerator(){};
//  OpGenerator spirvGroupNonUniformBroadcastGenerator(){};
//  OpGenerator spirvGroupNonUniformElectGenerator(){};
//  OpGenerator spirvGroupNonUniformFAddGenerator(){};
//  OpGenerator spirvGroupNonUniformFMaxGenerator(){};
//  OpGenerator spirvGroupNonUniformFMinGenerator(){};
//  OpGenerator spirvGroupNonUniformFMulGenerator(){};
//  OpGenerator spirvGroupNonUniformIAddGenerator(){};
//  OpGenerator spirvGroupNonUniformIMulGenerator(){};
//  OpGenerator spirvGroupNonUniformSMaxGenerator(){};
//  OpGenerator spirvGroupNonUniformSMinGenerator(){};
//  OpGenerator spirvGroupNonUniformShuffleDownGenerator(){};
//  OpGenerator spirvGroupNonUniformShuffleGenerator(){};
//  OpGenerator spirvGroupNonUniformShuffleUpGenerator(){};
//  OpGenerator spirvGroupNonUniformShuffleXorGenerator(){};
//  OpGenerator spirvGroupNonUniformUMaxGenerator(){};
//  OpGenerator spirvGroupNonUniformUMinGenerator(){};
//  OpGenerator spirvGroupSMaxGenerator(){};
//  OpGenerator spirvGroupSMinGenerator(){};
//  OpGenerator spirvGroupUMaxGenerator(){};
//  OpGenerator spirvGroupUMinGenerator(){};
//  OpGenerator spirvIAddCarryGenerator(){};
//  OpGenerator spirvIAddGenerator(){};
//  OpGenerator spirvIMulGenerator(){};
//  OpGenerator spirvINTELConvertBF16ToFGenerator(){};
//  OpGenerator spirvINTELConvertFToBF16Generator(){};
//  OpGenerator spirvINTELJointMatrixLoadGenerator(){};
//  OpGenerator spirvINTELJointMatrixMadGenerator(){};
//  OpGenerator spirvINTELJointMatrixStoreGenerator(){};
//  OpGenerator spirvINTELJointMatrixWorkItemLengthGenerator(){};
//  OpGenerator spirvINTELSubgroupBlockReadGenerator(){};
//  OpGenerator spirvINTELSubgroupBlockWriteGenerator(){};
//  OpGenerator spirvISubBorrowGenerator(){};
//  OpGenerator spirvISubGenerator(){};
//  OpGenerator spirvImageDrefGatherGenerator(){};
//  OpGenerator spirvImageGenerator(){};
//  OpGenerator spirvImageQuerySizeGenerator(){};
//  OpGenerator spirvInBoundsPtrAccessChainGenerator(){};

OpGenerator spirvIsInfGenerator() {
  return getSPIRVFUnaryOpGenerator<spirv::IsInfOp>("spirv.IsInf");
}

OpGenerator spirvIsNanGenerator() {
  return getSPIRVFUnaryOpGenerator<spirv::IsNanOp>("spirv.IsNan");
}

// OpGenerator spirvKHRAssumeTrueGenerator(){};
// OpGenerator spirvKHRCooperativeMatrixLengthGenerator(){};
// OpGenerator spirvKHRCooperativeMatrixLoadGenerator(){};
// OpGenerator spirvKHRCooperativeMatrixStoreGenerator(){};
// OpGenerator spirvKHRSubgroupBallotGenerator(){};
// OpGenerator spirvLoadGenerator(){};

// OpGenerator spirvLoopGenerator(){};
// OpGenerator spirvMatrixTimesMatrixGenerator(){};
// OpGenerator spirvMatrixTimesScalarGenerator(){};
// OpGenerator spirvMemoryBarrierGenerator(){};
// OpGenerator spirvMergeGenerator(){};
// OpGenerator spirvModuleGenerator(){};
// OpGenerator spirvNVCooperativeMatrixLengthGenerator(){};
// OpGenerator spirvNVCooperativeMatrixLoadGenerator(){};
// OpGenerator spirvNVCooperativeMatrixMulAddGenerator(){};
// OpGenerator spirvNVCooperativeMatrixStoreGenerator(){};

// OpGenerator spirvOrderedGenerator(){};
// OpGenerator spirvPtrAccessChainGenerator(){};
// OpGenerator spirvPtrCastToGenericGenerator(){};
// OpGenerator spirvReferenceOfGenerator(){};
// OpGenerator spirvReturnGenerator(){};
// OpGenerator spirvReturnValueGenerator(){};
// OpGenerator spirvSConvertGenerator(){};
// OpGenerator spirvSDivGenerator(){};
// OpGenerator spirvSDotAccSatGenerator(){};
// OpGenerator spirvSDotGenerator(){};
// OpGenerator spirvSModGenerator(){};
// OpGenerator spirvSMulExtendedGenerator(){};

// TODO-matrix
// OpGenerator spirvSNegateGenerator() {
//  return getSPIRVIntUnaryOpGenerator<spirv::SNegateOp>("spirv.SNegate");
//}
// TODO: need spirv.struct as result type.
// OpGenerator spirvGLFrexpStructGenerator() {
//  return
//  getSPIRVFloatUnaryOpGenerator<spirv::GLFrexpStructOp>("spirv.GL.Frexp");
//}

// OpGenerator spirvSRemGenerator(){};
// OpGenerator spirvSUDotAccSatGenerator(){};
// OpGenerator spirvSUDotGenerator(){};
// OpGenerator spirvSelectGenerator(){};
// OpGenerator spirvSelectionGenerator(){};
// OpGenerator spirvShiftLeftLogicalGenerator(){};
// OpGenerator spirvShiftRightArithmeticGenerator(){};
// OpGenerator spirvShiftRightLogicalGenerator(){};
// OpGenerator spirvSpecConstantCompositeGenerator(){};
// OpGenerator spirvSpecConstantGenerator(){};
// OpGenerator spirvSpecConstantOperationGenerator(){};
// OpGenerator spirvStoreGenerator(){};
// OpGenerator spirvTransposeGenerator(){};
// OpGenerator spirvUConvertGenerator(){};
// OpGenerator spirvUDivGenerator(){};
// OpGenerator spirvUDotAccSatGenerator(){};
// OpGenerator spirvUDotGenerator(){};

// OpGenerator spirvUModGenerator(){};
// OpGenerator spirvUMulExtendedGenerator(){};
// OpGenerator spirvUndefGenerator(){};
// OpGenerator spirvUnreachableGenerator(){};
// OpGenerator spirvVariableGenerator(){};
// OpGenerator spirvVectorExtractDynamicGenerator(){};
// OpGenerator spirvVectorInsertDynamicGenerator(){};
// OpGenerator spirvVectorShuffleGenerator(){};
// OpGenerator spirvVectorTimesScalarGenerator(){};
// OpGenerator spirvYieldGenerator(){};
