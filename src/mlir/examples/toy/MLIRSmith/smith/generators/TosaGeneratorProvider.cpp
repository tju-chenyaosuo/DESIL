//
// Created by Chenyao Suo on 2023/7/12.
//
#include "include/smith/RegionGeneration.h"
#include "include/smith/TypeGeneration.h"
#include "include/smith/generators/OpGeneration.h"

#include "mlir/Dialect/Tosa/IR/TosaOps.h"

using namespace mlir;

// Operator: tosa::AbsOp
// AbsOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState,
// ::mlir::Type output, ::mlir::Value input1)
OpGenerator tosaAbsGenerator() {
  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto operandCandidates = region.pool.searchCandidatesFrom(
        {PoolType::StaticShapedTensor, PoolType::DynamicShapedTensor},
        emptyFilter());
    if (operandCandidates.empty()) {
      operandCandidates.push_back(region.pool.generateRankedTensor(
          builder, loc, randomRankedTensorType(builder.getContext())));
    }
    auto src = sampleTypedValueFrom(operandCandidates, "tosa.abs");
    auto output = src.type;
    auto input1 = src.val;
    auto res_tensor = builder.create<mlir::tosa::AbsOp>(loc, output, input1);

    region.pool.addRankedTensor(
        TypeValue(mlir::dyn_cast<RankedTensorType>(src.type), res_tensor),
        "tosa.abs");
    return res_tensor.getOperation();
  };
}

// Operator: mlir::tosa::AddOp
// Documentation:
// input1	tensor of number values
// input2	tensor of number values
// output	tensor of number values
// Builder Definition File: ./include/mlir/Dialect/Tosa/IR/TosaOps.td
// Selected Builder Function:
// void AddOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState
// &odsState,
//                   ::mlir::Type output, ::mlir::Value input1, ::mlir::Value
//                   input2);
OpGenerator tosaAddGenerator() {
  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    // prepare operand
    auto operandCandidates = region.pool.searchCandidatesFrom(
        {PoolType::StaticShapedTensor, PoolType::DynamicShapedTensor},
        emptyFilter());
    if (operandCandidates.empty()) {
      operandCandidates.push_back(region.pool.generateRankedTensor(
          builder, loc, randomRankedTensorType(builder.getContext())));
    }
    auto src1 = sampleTypedValueFrom(operandCandidates, "tosa.add");
    auto src2 = region.pool.generateRankedTensor(
        builder, loc, mlir::dyn_cast<RankedTensorType>(src1.type));

    auto input1 = src1.val;
    auto input2 = src2.val;
    auto output = src1.type;

    auto res_tensor =
        builder.create<mlir::tosa::AddOp>(loc, output, input1, input2);

    region.pool.addRankedTensor(
        TypeValue(mlir::dyn_cast<RankedTensorType>(output), res_tensor), "tosa.add");
    return res_tensor.getOperation();
  };
}

// Operator: mlir::tosa::ApplyScaleOp
// Documentation:
// double_round	::mlir::BoolAttr	bool attribute
// value	signless-integer-like
// multiplier	signless-integer-like
// shift	signless-integer-8-bit-like
// output	signless-integer-like
// Builder Definition File: ./include/mlir/Dialect/Tosa/IR/TosaOps.td
// Selected Builder Function:
// void ApplyScaleOp::build(::mlir::OpBuilder &odsBuilder,
// ::mlir::OperationState &odsState,
//                          ::mlir::Type output, ::mlir::Value value,
//                          ::mlir::Value multiplier, ::mlir::Value shift, bool
//                          double_round)
OpGenerator tosaApplyScaleGenerator() {
  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    // prepare the parameters.
    auto candidates = searchIntegerCandidatesFrom(region.pool.intOrFloatPool);
    if (candidates.empty()) {
      candidates.push_back(
          region.pool.generateInteger(builder, loc, builder.getI32Type()));
    }
    auto value_vt = sampleTypedValueFrom(candidates, "tosa.applyScale");
    auto multiplier_vt = sampleTypedValueFrom(candidates, "tosa.applyScale");
    auto shift_vt =
        region.pool.generateInteger(builder, loc, builder.getI8Type());
    auto value = value_vt.val;
    auto multiplier = multiplier_vt.val;
    auto shift = shift_vt.val;
    bool double_round = UR(2);
    auto output = mlir::dyn_cast<IntegerType>(value_vt.type);

    // create instruction.
    auto res = builder.create<mlir::tosa::ApplyScaleOp>(
        loc, output, value, multiplier, shift, double_round);
    region.pool.addIntOrFloat(TypeValue(output, res), "tosa.applyScale");
    return res.getOperation();
  };
}

// Operator:tosa.argmax (mlir::tosa::ArgMaxOp)
// export PATH=/data/hqy/mlirsmith-dev/build/tools/llvm-symbolizer:$PATH

// axis	::mlir::IntegerAttr	64-bit signless integer attribute
// input	tensor of number values
// output	tensor of number values

// include/mlir/Dialect/Tosa/IR/TosaOps.td
// void ArgMaxOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState
// &odsState, ::mlir::Type output, ::mlir::Value input, uint64_t axis) void
// ArgMaxOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState
// &odsState, ::mlir::Type output, ::mlir::Value input, ::mlir::IntegerAttr
// axis)
OpGenerator tosaArgMaxGenerator() {
  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto operandCandidates = region.pool.searchCandidatesFrom(
        {PoolType::StaticShapedTensor, PoolType::DynamicShapedTensor},
        [](TypeValue tval) {
          return mlir::dyn_cast<RankedTensorType>(tval.type).getRank() > 0 &&
                 mlir::dyn_cast<RankedTensorType>(tval.type).getRank() <= 4;
        });

    if (operandCandidates.empty()) {
      SmallVector<int64_t> new_shape;
      new_shape.push_back(1);
      for (int i = 0; i < UR(3); i++) {
        new_shape.push_back(dimPool[UR(dimPool.size())]);
      }
      operandCandidates.push_back(region.pool.generateRankedTensor(
          builder, loc,
          RankedTensorType::get(new_shape,
                                randomIntOrFloatType(builder.getContext()))));
    }

    auto src = sampleTypedValueFrom(operandCandidates, "tosa.argmax");
    auto input = src.val;
    long long axis = UR(mlir::dyn_cast<RankedTensorType>(input.getType()).getRank());

    SmallVector<int64_t> shape;
    auto srcTy = mlir::dyn_cast<RankedTensorType>(src.type);

    if (srcTy.hasStaticShape()) {
      for (uint64_t i = 0; i < srcTy.getRank(); ++i) {
        if (i == axis)
          continue;
        shape.push_back(ShapedType::kDynamic);
      }
    } else {
      for (uint64_t i = 0; i < srcTy.getRank(); ++i) {
        if (i == axis)
          continue;
        if (srcTy.isDynamicDim(i)) {
          shape.push_back(dimPool[UR(dimPool.size())]);
        } else {
          shape.push_back(srcTy.getDimSize(i));
        }
      }
    }

    auto output = RankedTensorType::get(shape, srcTy.getElementType());

    auto res = builder.create<mlir::tosa::ArgMaxOp>(loc, output, input, axis);
    region.pool.addRankedTensor(TypeValue(output, res), "tosa.argmax");
    return res.getOperation();
  };
}

/*
tosa.arithmetic_right_shift (mlir::tosa::ArithmeticRightShiftOp)
Elementwise arithmetic right shift of input1 by the amount specified in input2.
Axis of size 1 will be broadcast, as necessary.

round	::mlir::BoolAttr	bool attribute
input1	tensor of number values
input2	tensor of number values
output	tensor of number values
void ArithmeticRightShiftOp::build(::mlir::OpBuilder &odsBuilder,
::mlir::OperationState &odsState,
::mlir::Type output, ::mlir::Value input1, ::mlir::Value input2, bool round)
*/
OpGenerator tosaArithmeticRightShiftGenerator() {
  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto operandCandidates = region.pool.searchCandidatesFrom(
        {PoolType::StaticShapedTensor, PoolType::DynamicShapedTensor},
        emptyFilter());

    if (operandCandidates.empty()) {
      operandCandidates.push_back(region.pool.generateRankedTensor(
          builder, loc, randomRankedTensorType(builder.getContext())));
    }
    auto src =
        sampleTypedValueFrom(operandCandidates, "tosa.arithmetic_right_shift");

    auto input1 = src.val;
    auto input2 = src.val;
    auto output = mlir::dyn_cast<RankedTensorType>(src.type);
    bool round = UR(2);

    auto res = builder.create<mlir::tosa::ArithmeticRightShiftOp>(
        loc, output, input1, input2, round);
    region.pool.addRankedTensor(TypeValue(output, res),
                                "tosa.arithmetic_right_shift");
    return res.getOperation();
  };
}

/*tosa.avg_pool2d (mlir::tosa::AvgPool2dOp) ¶
Performs max pooling on the input.
This performs an average pooling over the given input tensor. A sliding window
of size given by is passed over the input tensor, with the mean value being
placed in the output tensor.

kernel	::mlir::DenseI64ArrayAttr	i64 dense array attribute with exactly 2
elements stride	::mlir::DenseI64ArrayAttr	i64 dense array attribute with
exactly 2 elements pad	::mlir::DenseI64ArrayAttr	i64 dense array
attribute with exactly 4 elements acc_type	::mlir::TypeAttr	type
attribute of 32-bit signless integer or 32-bit signed integer or 16-bit float or
32-bit float quantization_info	mlir::tosa::UnaryOpQuantizationAttr
Attribute for UnaryOp quantization information. input	unranked tensor of
number values or 4D tensor of number values output	unranked tensor of
number values or 4D tensor of number values void
AvgPool2dOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState
&odsState, ::mlir::Type outputType, ::mlir::Value input,
::mlir::DenseI64ArrayAttr kernel, ::mlir::DenseI64ArrayAttr s tride,
::mlir::DenseI64ArrayAttr pad)
*/

/*tosa.bitwise_and (mlir::tosa::BitwiseAndOp)
Bitwise AND operator
Elementwise bitwise AND of input1 and input2. Axis of size 1 will be broadcast
as necessary.

input1	tensor of number values
input2	tensor of number values
output	tensor of number values

void BitwiseAndOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState
&odsState, ::mlir::Type output, ::mlir::Value input1, ::mlir::Value input2)
*/
OpGenerator tosaBitwiseAndGenerator() {
  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto operandCandidates = region.pool.searchCandidatesFrom(
        {PoolType::StaticShapedTensor, PoolType::DynamicShapedTensor},
        emptyFilter());

    if (operandCandidates.empty()) {
      operandCandidates.push_back(region.pool.generateRankedTensor(
          builder, loc, randomRankedTensorType(builder.getContext())));
    }
    auto src = sampleTypedValueFrom(operandCandidates, "tosa.bitwise_and");

    auto input1 = src.val;
    auto input2 = src.val;
    auto output = mlir::dyn_cast<RankedTensorType>(src.type);
    auto res =
        builder.create<mlir::tosa::BitwiseAndOp>(loc, output, input1, input2);
    region.pool.addRankedTensor(TypeValue(output, res), "tosa.bitwise_and");
    return res.getOperation();
  };
}

/*tosa.bitwise_not (mlir::tosa::BitwiseNotOp)

input1	tensor of number values
output	tensor of number values

*/
OpGenerator tosaBitwiseNotGenerator() {
  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto operandCandidates = region.pool.searchCandidatesFrom(
        {PoolType::StaticShapedTensor, PoolType::DynamicShapedTensor},
        emptyFilter());

    if (operandCandidates.empty()) {
      operandCandidates.push_back(region.pool.generateRankedTensor(
          builder, loc, randomRankedTensorType(builder.getContext())));
    }
    auto src = sampleTypedValueFrom(operandCandidates, "tosa.bitwise_not");
    auto input = src.val;
    auto output = mlir::dyn_cast<RankedTensorType>(src.type);
    auto res = builder.create<mlir::tosa::BitwiseNotOp>(loc, output, input);
    region.pool.addRankedTensor(TypeValue(output, res), "tosa.bitwise_not");
    return res.getOperation();
  };
}

/*tosa.bitwise_or (mlir::tosa::BitwiseOrOp)
input1	tensor of number values
input2	tensor of number values
output	tensor of number values

void BitwiseOrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState
&odsState, ::mlir::Type output, ::mlir::Value input1, ::mlir::Value input2)
*/
OpGenerator tosaBitwiseOrGenerator() {
  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto operandCandidates = region.pool.searchCandidatesFrom(
        {PoolType::StaticShapedTensor, PoolType::DynamicShapedTensor},
        emptyFilter());

    if (operandCandidates.empty()) {
      operandCandidates.push_back(region.pool.generateRankedTensor(
          builder, loc, randomRankedTensorType(builder.getContext())));
    }
    auto src = sampleTypedValueFrom(operandCandidates, "tosa.bitwise_or");

    auto input1 = src.val;
    auto input2 = src.val;
    auto output = mlir::dyn_cast<RankedTensorType>(src.type);
    auto res =
        builder.create<mlir::tosa::BitwiseOrOp>(loc, output, input1, input2);
    region.pool.addRankedTensor(TypeValue(output, res), "tosa.bitwise_or");
    return res.getOperation();
  };
}

/*tosa.bitwise_xor (mlir::tosa::BitwiseXorOp)
input1	tensor of number values
input2	tensor of number values
output	tensor of number values
void BitwiseXorOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState
&odsState, ::mlir::Type output, ::mlir::Value input1, ::mlir::Value input2)
*/
OpGenerator tosaBitwiseXorGenerator() {
  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto operandCandidates = region.pool.searchCandidatesFrom(
        {PoolType::StaticShapedTensor, PoolType::DynamicShapedTensor},
        emptyFilter());

    if (operandCandidates.empty()) {
      operandCandidates.push_back(region.pool.generateRankedTensor(
          builder, loc, randomRankedTensorType(builder.getContext())));
    }
    auto src = sampleTypedValueFrom(operandCandidates, "tosa.bitwise_xor");

    auto input1 = src.val;
    auto input2 = src.val;
    auto output = mlir::dyn_cast<RankedTensorType>(src.type);
    auto res =
        builder.create<mlir::tosa::BitwiseXorOp>(loc, output, input1, input2);
    region.pool.addRankedTensor(TypeValue(output, res), "tosa.bitwise_xor");
    return res.getOperation();
  };
}

/*tosa.cast (mlir::tosa::CastOp)
void CastOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState
&odsState, ::mlir::Type output, ::mlir::Value input) Performs a set of
permissible cast operations

input:tensor of number_plus_f64 values
output	tensor of number_plus_f64 values
void CastOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState
&odsState, ::mlir::Type output, ::mlir::Value input)
*/

OpGenerator tosaCastGenerator() {
  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto operandCandidates = region.pool.searchCandidatesFrom(
        {PoolType::StaticShapedTensor, PoolType::DynamicShapedTensor},
        [&](TypeValue typeValue) {
          return mlir::dyn_cast<TensorType>(typeValue.type).getElementType() ==
                 builder.getI8Type();
        });

    if (operandCandidates.empty()) {
      SmallVector<int64_t> shape;

      for (int i = 0; i < UR(dimPool[UR(dimPool.size())]); i++) {
        shape.push_back(dimPool[UR(dimPool.size())]);
      }
      operandCandidates.push_back(region.pool.generateRankedTensor(
          builder, loc, RankedTensorType::get(shape, builder.getI8Type())));
    }

    auto src = sampleTypedValueFrom(operandCandidates, "tosa.cast");

    auto input = src.val;
    auto shape = mlir::dyn_cast<RankedTensorType>(src.type).getShape();
    auto output = RankedTensorType::get(shape, builder.getI16Type());

    auto res = builder.create<mlir::tosa::CastOp>(loc, output, input);
    region.pool.addRankedTensor(TypeValue(output, res), "tosa.cast");
    return res.getOperation();
  };
}

/*tosa.ceil (mlir::tosa::CeilOp)
:Elementwise ceiling operation
input1	tensor of number values
output	tensor of number values
void CeilOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState
&odsState, ::mlir::Type output, ::mlir::Value input1)
*/
OpGenerator tosaCeilGenerator() {
  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto operandCandidates = region.pool.searchCandidatesFrom(
        {PoolType::StaticShapedTensor, PoolType::DynamicShapedTensor},
        emptyFilter());

    if (operandCandidates.empty()) {
      operandCandidates.push_back(region.pool.generateRankedTensor(
          builder, loc, randomRankedTensorType(builder.getContext())));
    }
    auto src = sampleTypedValueFrom(operandCandidates, "tosa.ceil");

    auto input = src.val;
    auto output = mlir::dyn_cast<RankedTensorType>(src.type);
    auto res = builder.create<mlir::tosa::CeilOp>(loc, output, input);
    region.pool.addRankedTensor(TypeValue(output, res), "tosa.ceil");
    return res.getOperation();
  };
}

// tosa.clamp (mlir::tosa::ClampOp)
// void ClampOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState
// &odsState, ::mlir::Type output, ::mlir::Value input, uint64_t min_int,
// uint64_t max_int, ::llvm::APFloat min _fp, ::llvm::APFloat max_fp)
//  min_int	::mlir::IntegerAttr	64-bit signless integer attribute
//  max_int	::mlir::IntegerAttr	64-bit signless integer attribute
//  min_fp	::mlir::FloatAttr	32-bit float attribute
//  max_fp	::mlir::FloatAttr	32-bit float attribute

// OpGenerator tosaClampGenerator() {
//   return [](OpBuilder &builder, Location loc, OpRegion &region) {
//     auto operandCandidates = region.pool.searchCandidatesFrom(
//       {PoolType::StaticShapedTensor, PoolType::DynamicShapedTensor},
//       emptyFilter());
//     if (operandCandidates.empty()) {
//         operandCandidates.push_back(
//           region.pool.generateRankedTensor(builder, loc,
//           randomRankedTensorType(builder.getContext())));
//     }
//     auto src1 = sampleTypedValueFrom(operandCandidates, "tosa.clamp");

//     auto input = src1.val;
//     auto output = src1.type;
//     uint64_t max_int = ((unsigned long long)rand() << 32) | rand();
//     uint64_t min_int = (unsigned long long)UR(max_int+1);
//     // 创建一个随机数生成器
//     std::random_device rd;
//     std::mt19937 gen(rd());

//     // 定义随机数的分布范围
//     std::uniform_real_distribution<float> dist(0.0f, 1.0f);

//     // 生成随机数
//     ::llvm::APFloat min_fp = APFloat(::llvm::APFloat::IEEEsingle, dist(gen));
//     ::llvm::APFloat max_fp = APFloat(::llvm::APFloat::IEEEsingle, dist(gen));

//     auto res = builder.create<mlir::tosa::ClampOp>(loc, output,
//     input,min_int, max_int, min_fp, max_fp); region.pool.addRankedTensor(
//       TypeValue(mlir::dyn_cast<RankedTensorType>(output), res),
//       "tosa.clamp");
//     return res.getOperation();
//   };
// }

//  ::mlir::Type output, ::mlir::Value input1) {
// tosa.identity (mlir::tosa::IdentityOp)
OpGenerator tosaIdentityOpGenerator() {
  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto operandCandidates = region.pool.searchCandidatesFrom(
        {PoolType::StaticShapedTensor, PoolType::DynamicShapedTensor},
        emptyFilter());
    if (operandCandidates.empty()) {
      operandCandidates.push_back(region.pool.generateRankedTensor(
          builder, loc, randomRankedTensorType(builder.getContext())));
    }
    auto src1 = sampleTypedValueFrom(operandCandidates, "tosa.identity");

    auto input1 = src1.val;
    auto output = src1.type;

    auto res = builder.create<mlir::tosa::IdentityOp>(loc, output, input1);
    region.pool.addRankedTensor(
        TypeValue(mlir::dyn_cast<RankedTensorType>(output), res), "tosa.identity");
    return res.getOperation();
  };
}
// ::mlir::TypeRange output, ::mlir::Value cond, ::mlir::ValueRange inputs) {
// tosa.cond_if (mlir::tosa::IfOp)

// ::mlir::Type output, ::mlir::Value input1) {
// tosa.log (mlir::tosa::LogOp)
OpGenerator tosaLogOpGenerator() {
  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto operandCandidates = region.pool.searchCandidatesFrom(
        {PoolType::StaticShapedTensor, PoolType::DynamicShapedTensor},
        emptyFilter());
    if (operandCandidates.empty()) {
      operandCandidates.push_back(region.pool.generateRankedTensor(
          builder, loc, randomRankedTensorType(builder.getContext())));
    }
    auto src1 = sampleTypedValueFrom(operandCandidates, "tosa.log");

    auto input1 = src1.val;
    auto output = src1.type;

    auto res = builder.create<mlir::tosa::LogOp>(loc, output, input1);
    region.pool.addRankedTensor(
        TypeValue(mlir::dyn_cast<RankedTensorType>(output), res), "tosa.log");
    return res.getOperation();
  };
}

// ::mlir::Type z, ::mlir::Value input1, ::mlir::Value input2) {
// tosa.logical_and (mlir::tosa::LogicalAndOp)
OpGenerator tosaLogicalAndOpGenerator() {
  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    // prepare operand
    auto operandCandidates = region.pool.searchCandidatesFrom(
        {PoolType::StaticShapedTensor, PoolType::DynamicShapedTensor},
        [&](TypeValue tVal) {
          return mlir::dyn_cast<RankedTensorType>(tVal.type).getElementType() ==
                 builder.getI1Type();
        });
    if (operandCandidates.empty()) {
      SmallVector<int64_t> newShape;
      auto predTy = RankedTensorType::get(newShape, builder.getI1Type());
      auto dest = region.pool.generateRankedTensor(builder, loc, predTy);
      operandCandidates.push_back(dest);
    }
    auto src1 = sampleTypedValueFrom(operandCandidates, "tosa.logical_and");
    auto src2 = region.pool.generateRankedTensor(
        builder, loc, mlir::dyn_cast<RankedTensorType>(src1.type));

    auto input1 = src1.val;
    auto input2 = src2.val;
    auto z = src1.type;

    auto res_tensor =
        builder.create<mlir::tosa::LogicalAndOp>(loc, z, input1, input2);

    region.pool.addRankedTensor(
        TypeValue(mlir::dyn_cast<RankedTensorType>(z), res_tensor),
        "tosa.logical_and");
    return res_tensor.getOperation();
  };
}

// ::mlir::Type output, ::mlir::Value input1, ::mlir::Value input2) {
// tosa.logical_left_shift (mlir::tosa::LogicalLeftShiftOp)
OpGenerator tosaLogicalLeftShiftOpGenerator() {
  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    // prepare operand
    auto operandCandidates = region.pool.searchCandidatesFrom(
        {PoolType::StaticShapedTensor, PoolType::DynamicShapedTensor},
        emptyFilter());
    if (operandCandidates.empty()) {
      operandCandidates.push_back(region.pool.generateRankedTensor(
          builder, loc, randomRankedTensorType(builder.getContext())));
    }
    auto src1 =
        sampleTypedValueFrom(operandCandidates, "tosa.logical_left_shift");
    auto src2 = region.pool.generateRankedTensor(
        builder, loc, mlir::dyn_cast<RankedTensorType>(src1.type));

    auto input1 = src1.val;
    auto input2 = src2.val;
    auto output = src1.type;

    auto res_tensor = builder.create<mlir::tosa::LogicalLeftShiftOp>(
        loc, output, input1, input2);

    region.pool.addRankedTensor(
        TypeValue(mlir::dyn_cast<RankedTensorType>(output), res_tensor),
        "tosa.logical_left_shift");
    return res_tensor.getOperation();
  };
}
// ::mlir::Type output, ::mlir::Value input1) {
// tosa.logical_not (mlir::tosa::LogicalNotOp)
OpGenerator tosaLogicalNotOpGenerator() {
  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto operandCandidates = region.pool.searchCandidatesFrom(
        {PoolType::StaticShapedTensor, PoolType::DynamicShapedTensor},
        [&](TypeValue tVal) {
          return mlir::dyn_cast<RankedTensorType>(tVal.type).getElementType() ==
                 builder.getI1Type();
        });
    if (operandCandidates.empty()) {
      SmallVector<int64_t> newShape;
      auto predTy = RankedTensorType::get(newShape, builder.getI1Type());
      auto dest = region.pool.generateRankedTensor(builder, loc, predTy);
      operandCandidates.push_back(dest);
    }
    auto src1 = sampleTypedValueFrom(operandCandidates, "tosa.logical_not");

    auto input1 = src1.val;
    auto output = src1.type;

    auto res = builder.create<mlir::tosa::LogicalNotOp>(loc, output, input1);
    region.pool.addRankedTensor(
        TypeValue(mlir::dyn_cast<RankedTensorType>(output), res),
        "tosa.logical_not");
    return res.getOperation();
  };
}

// ::mlir::Type z, ::mlir::Value input1, ::mlir::Value input2) {
// tosa.logical_or (mlir::tosa::LogicalOrOp)
OpGenerator tosaLogicalOrOpGenerator() {
  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    // prepare operand
    auto operandCandidates = region.pool.searchCandidatesFrom(
        {PoolType::StaticShapedTensor, PoolType::DynamicShapedTensor},
        [&](TypeValue tVal) {
          return mlir::dyn_cast<RankedTensorType>(tVal.type).getElementType() ==
                 builder.getI1Type();
        });
    if (operandCandidates.empty()) {
      SmallVector<int64_t> newShape;
      auto predTy = RankedTensorType::get(newShape, builder.getI1Type());
      auto dest = region.pool.generateRankedTensor(builder, loc, predTy);
      operandCandidates.push_back(dest);
    }
    auto src1 = sampleTypedValueFrom(operandCandidates, "tosa.logical_or");
    auto src2 = region.pool.generateRankedTensor(
        builder, loc, mlir::dyn_cast<RankedTensorType>(src1.type));

    auto input1 = src1.val;
    auto input2 = src2.val;
    auto z = src1.type;

    auto res_tensor =
        builder.create<mlir::tosa::LogicalOrOp>(loc, z, input1, input2);

    region.pool.addRankedTensor(
        TypeValue(mlir::dyn_cast<RankedTensorType>(z), res_tensor),
        "tosa.logical_or");
    return res_tensor.getOperation();
  };
}
// ::mlir::Type output, ::mlir::Value input1, ::mlir::Value input2) {
// tosa.logical_right_shift (mlir::tosa::LogicalRightShiftOp)
OpGenerator tosaLogicalRightShiftOpGenerator() {
  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    // prepare operand
    auto operandCandidates = region.pool.searchCandidatesFrom(
        {PoolType::StaticShapedTensor, PoolType::DynamicShapedTensor},
        emptyFilter());
    if (operandCandidates.empty()) {
      operandCandidates.push_back(region.pool.generateRankedTensor(
          builder, loc, randomRankedTensorType(builder.getContext())));
    }
    auto src1 =
        sampleTypedValueFrom(operandCandidates, "tosa.logical_right_shift");
    auto src2 = region.pool.generateRankedTensor(
        builder, loc, mlir::dyn_cast<RankedTensorType>(src1.type));

    auto input1 = src1.val;
    auto input2 = src2.val;
    auto output = src1.type;

    auto res_tensor = builder.create<mlir::tosa::LogicalRightShiftOp>(
        loc, output, input1, input2);

    region.pool.addRankedTensor(
        TypeValue(mlir::dyn_cast<RankedTensorType>(output), res_tensor),
        "tosa.logical_right_shift");
    return res_tensor.getOperation();
  };
}
// ::mlir::Type z, ::mlir::Value input1, ::mlir::Value input2) {
// tosa.logical_xor (mlir::tosa::LogicalXorOp)
OpGenerator tosaLogicalXorOpGenerator() {
  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    // prepare operand
    auto operandCandidates = region.pool.searchCandidatesFrom(
        {PoolType::StaticShapedTensor, PoolType::DynamicShapedTensor},
        [&](TypeValue tVal) {
          return mlir::dyn_cast<RankedTensorType>(tVal.type).getElementType() ==
                 builder.getI1Type();
        });
    if (operandCandidates.empty()) {
      SmallVector<int64_t> newShape;
      auto predTy = RankedTensorType::get(newShape, builder.getI1Type());
      auto dest = region.pool.generateRankedTensor(builder, loc, predTy);
      operandCandidates.push_back(dest);
    }
    auto src1 = sampleTypedValueFrom(operandCandidates, "tosa.logical_xor");
    auto src2 = region.pool.generateRankedTensor(
        builder, loc, mlir::dyn_cast<RankedTensorType>(src1.type));

    auto input1 = src1.val;
    auto input2 = src2.val;
    auto z = src1.type;

    auto res_tensor =
        builder.create<mlir::tosa::LogicalXorOp>(loc, z, input1, input2);

    region.pool.addRankedTensor(
        TypeValue(mlir::dyn_cast<RankedTensorType>(z), res_tensor),
        "tosa.logical_xor");
    return res_tensor.getOperation();
  };
}

// tosa.matmul (mlir::tosa::MatMulOp)
// Type outputType, Value a, Value b) {
OpGenerator tosaMatMulOpGenerator() {
  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto operandCandidates = region.pool.searchCandidatesFrom(
        {PoolType::StaticShapedTensor, PoolType::DynamicShapedTensor},
        [](TypeValue tval) {
          return mlir::dyn_cast<RankedTensorType>(tval.type).getRank() == 3;
        });
    if (operandCandidates.empty()) {
      SmallVector<int64_t> newShape;
      for (int i = 0; i < 3; i++) {
        newShape.push_back(1);
      }
      auto destTy = RankedTensorType::get(
          newShape, randomIntOrFloatType(builder.getContext()));
      auto dest = region.pool.generateRankedTensor(builder, loc, destTy);
      operandCandidates.push_back(dest);
    }

    TypeValue src1 = sampleTypedValueFrom(operandCandidates, "tosa.matmul");
    TypeValue src2 = region.pool.generateRankedTensor(
        builder, loc, mlir::dyn_cast<RankedTensorType>(src1.type));

    Type outputType = src1.type;
    Value a = src1.val;
    Value b = src2.val;

    auto res = builder.create<mlir::tosa::MatMulOp>(loc, outputType, a, b);
    region.pool.addRankedTensor(
        TypeValue(mlir::dyn_cast<RankedTensorType>(outputType), res), "tosa.matmul");
    return res.getOperation();
  };
}

// tosa.max_pool2d (mlir::tosa::MaxPool2dOp)
// ::mlir::Type output, ::mlir::Value input,
// ::mlir::DenseI64ArrayAttr kernel, ::mlir::DenseI64ArrayAttr stride,
// ::mlir::DenseI64ArrayAttr pad) {
OpGenerator tosaMaxPool2dOpGenerator() {
  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto operandCandidates = region.pool.searchCandidatesFrom(
        {PoolType::StaticShapedTensor, PoolType::DynamicShapedTensor},
        [](TypeValue tval) {
          return mlir::dyn_cast<RankedTensorType>(tval.type).getRank() == 4;
        });
    if (operandCandidates.empty()) {
      SmallVector<int64_t> newShape;
      for (int i = 0; i < 4; i++) {
        newShape.push_back(1);
      }
      auto destTy = RankedTensorType::get(
          newShape, randomIntOrFloatType(builder.getContext()));
      auto dest = region.pool.generateRankedTensor(builder, loc, destTy);
      operandCandidates.push_back(dest);
    }

    auto src1 = sampleTypedValueFrom(operandCandidates, "tosa.max_pool2d");
    auto input = src1.val;
    auto output = src1.type;

    SmallVector<int64_t> a, b, c;
    for (int i = 0; i < 2; i++) {
      a.push_back(1);
    }
    for (int i = 0; i < 2; i++) {
      b.push_back(1);
    }
    for (int i = 0; i < 4; i++) {
      c.push_back(1);
    }
    auto kernel = builder.getDenseI64ArrayAttr(a);
    auto stride = builder.getDenseI64ArrayAttr(b);
    auto pad = builder.getDenseI64ArrayAttr(c);

    auto res = builder.create<mlir::tosa::MaxPool2dOp>(loc, output, input,
                                                       kernel, stride, pad);
    region.pool.addRankedTensor(
        TypeValue(mlir::dyn_cast<RankedTensorType>(output), res), "tosa.max_pool2d");
    return res.getOperation();
  };
}

// tosa.maximum (mlir::tosa::MaximumOp)
// ::mlir::Type output, ::mlir::Value input1, ::mlir::Value input2) {
OpGenerator tosaMaximumOpGenerator() {
  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    // prepare operand
    auto operandCandidates = region.pool.searchCandidatesFrom(
        {PoolType::StaticShapedTensor, PoolType::DynamicShapedTensor},
        emptyFilter());
    if (operandCandidates.empty()) {
      operandCandidates.push_back(region.pool.generateRankedTensor(
          builder, loc, randomRankedTensorType(builder.getContext())));
    }
    auto src1 = sampleTypedValueFrom(operandCandidates, "tosa.maximum");
    auto src2 = region.pool.generateRankedTensor(
        builder, loc, mlir::dyn_cast<RankedTensorType>(src1.type));

    auto input1 = src1.val;
    auto input2 = src2.val;
    auto output = src1.type;

    auto res_tensor =
        builder.create<mlir::tosa::MaximumOp>(loc, output, input1, input2);

    region.pool.addRankedTensor(
        TypeValue(mlir::dyn_cast<RankedTensorType>(output), res_tensor),
        "tosa.maximum");
    return res_tensor.getOperation();
  };
}

// tosa.minimum (mlir::tosa::MinimumOp)
OpGenerator tosaMinimumOpGenerator() {
  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    // prepare operand
    auto operandCandidates = region.pool.searchCandidatesFrom(
        {PoolType::StaticShapedTensor, PoolType::DynamicShapedTensor},
        emptyFilter());
    if (operandCandidates.empty()) {
      operandCandidates.push_back(region.pool.generateRankedTensor(
          builder, loc, randomRankedTensorType(builder.getContext())));
    }
    auto src1 = sampleTypedValueFrom(operandCandidates, "tosa.minimum");
    auto src2 = region.pool.generateRankedTensor(
        builder, loc, mlir::dyn_cast<RankedTensorType>(src1.type));

    auto input1 = src1.val;
    auto input2 = src2.val;
    auto output = src1.type;

    auto res_tensor =
        builder.create<mlir::tosa::MinimumOp>(loc, output, input1, input2);

    region.pool.addRankedTensor(
        TypeValue(mlir::dyn_cast<RankedTensorType>(output), res_tensor),
        "tosa.minimum");
    return res_tensor.getOperation();
  };
}

// tosa.mul (mlir::tosa::MulOp)
// ::mlir::Type output, ::mlir::Value input1, ::mlir::Value input2,
// ::mlir::IntegerAttr shift) {
OpGenerator tosaMulOpGenerator() {
  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto operandCandidates = region.pool.searchCandidatesFrom(
        {PoolType::StaticShapedTensor, PoolType::DynamicShapedTensor},
        emptyFilter());
    if (operandCandidates.empty()) {
      operandCandidates.push_back(region.pool.generateRankedTensor(
          builder, loc, randomRankedTensorType(builder.getContext())));
    }
    auto src1 = sampleTypedValueFrom(operandCandidates, "tosa.mul");
    auto src2 = region.pool.generateRankedTensor(
        builder, loc, mlir::dyn_cast<RankedTensorType>(src1.type));

    auto output = src1.type;
    auto input1 = src1.val;
    auto input2 = src2.val;
    auto shift = builder.getIntegerAttr(builder.getI32Type(),
                                        ((long long)rand() << 32) | rand());

    auto res =
        builder.create<mlir::tosa::MulOp>(loc, output, input1, input2, shift);
    region.pool.addRankedTensor(
        TypeValue(mlir::dyn_cast<RankedTensorType>(output), res), "tosa.mul");
    return res.getOperation();
  };
}

// tosa.negate (mlir::tosa::NegateOp)
// Type outputType, Value input) {
OpGenerator tosaNegateOpGenerator() {
  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto operandCandidates = region.pool.searchCandidatesFrom(
        {PoolType::StaticShapedTensor, PoolType::DynamicShapedTensor},
        emptyFilter());

    if (operandCandidates.empty()) {
      operandCandidates.push_back(region.pool.generateRankedTensor(
          builder, loc, randomRankedTensorType(builder.getContext())));
    }

    TypeValue src1 = sampleTypedValueFrom(operandCandidates, "tosa.negate");

    Type outputType = src1.type;
    Value input = src1.val;

    auto res = builder.create<mlir::tosa::NegateOp>(loc, outputType, input);
    region.pool.addRankedTensor(
        TypeValue(mlir::dyn_cast<RankedTensorType>(outputType), res), "tosa.negate");
    return res.getOperation();
  };
}

// tosa.pad (mlir::tosa::PadOp)
// Type outputType, Value input, Value paddings) {
OpGenerator tosaPadOpGenerator() {
  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto operandCandidates = region.pool.searchCandidatesFrom(
        {PoolType::StaticShapedTensor, PoolType::DynamicShapedTensor},
        emptyFilter());

    if (operandCandidates.empty()) {
      operandCandidates.push_back(region.pool.generateRankedTensor(
          builder, loc, randomRankedTensorType(builder.getContext())));
    }
    auto paddingsCandidates = region.pool.searchCandidatesFrom(
        {PoolType::StaticShapedTensor, PoolType::DynamicShapedTensor},
        [&](TypeValue tval) {
          return mlir::dyn_cast<RankedTensorType>(tval.type).getElementType() ==
                     builder.getI64Type() ||
                 mlir::dyn_cast<RankedTensorType>(tval.type).getElementType() ==
                     builder.getI32Type();
        });

    if (paddingsCandidates.empty()) {
      SmallVector<int64_t> newShape;
      for (int i = 0; i < UR(5); i++) {
        newShape.push_back(1);
      }
      paddingsCandidates.push_back(region.pool.generateRankedTensor(
          builder, loc, RankedTensorType::get(newShape, builder.getI64Type())));
    }

    TypeValue src1 = sampleTypedValueFrom(operandCandidates, "tosa.pad");
    TypeValue src2 = sampleTypedValueFrom(paddingsCandidates, "tosa.pad");

    Type outputType = src1.type;
    Value input = src1.val;
    Value paddings = src2.val;

    auto res =
        builder.create<mlir::tosa::PadOp>(loc, outputType, input, paddings);
    region.pool.addRankedTensor(
        TypeValue(mlir::dyn_cast<RankedTensorType>(outputType), res), "tosa.pad");
    return res.getOperation();
  };
}

// tosa.pow (mlir::tosa::PowOp)
// ::mlir::Type z, ::mlir::Value input1, ::mlir::Value input2) {
OpGenerator tosaPowOpGenerator() {
  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    // prepare operand
    auto operandCandidates = region.pool.searchCandidatesFrom(
        {PoolType::StaticShapedTensor, PoolType::DynamicShapedTensor},
        emptyFilter());
    if (operandCandidates.empty()) {
      operandCandidates.push_back(region.pool.generateRankedTensor(
          builder, loc, randomRankedTensorType(builder.getContext())));
    }
    auto src1 = sampleTypedValueFrom(operandCandidates, "tosa.pow");
    auto src2 = region.pool.generateRankedTensor(
        builder, loc, mlir::dyn_cast<RankedTensorType>(src1.type));

    auto input1 = src1.val;
    auto input2 = src2.val;
    auto z = src1.type;

    auto res_tensor = builder.create<mlir::tosa::PowOp>(loc, z, input1, input2);

    region.pool.addRankedTensor(
        TypeValue(mlir::dyn_cast<RankedTensorType>(z), res_tensor), "tosa.pow");
    return res_tensor.getOperation();
  };
}

// ::mlir::Type output_real, ::mlir::Type output_imag, ::mlir::Value input) {
// tosa.rfft2d (mlir::tosa::RFFT2dOp)
OpGenerator tosaRFFT2dOpGenerator() {
  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    // prepare operand
    auto operandCandidates = region.pool.searchCandidatesFrom(
        {PoolType::StaticShapedTensor, PoolType::DynamicShapedTensor},
        [](TypeValue tval) {
          return mlir::dyn_cast<RankedTensorType>(tval.type).getRank() == 3;
        });
    if (operandCandidates.empty()) {
      SmallVector<int64_t> newShape;
      for (int i = 0; i < 3; i++) {
        newShape.push_back(1);
      }
      auto destTy = RankedTensorType::get(
          newShape, randomIntOrFloatType(builder.getContext()));
      auto dest = region.pool.generateRankedTensor(builder, loc, destTy);
      operandCandidates.push_back(dest);
    }

    auto src1 = sampleTypedValueFrom(operandCandidates, "tosa.rfft2d");
    auto src2 = region.pool.generateRankedTensor(
        builder, loc, mlir::dyn_cast<RankedTensorType>(src1.type));

    auto output_real = src1.type;
    auto output_imag = src2.type;
    auto input = src2.val;

    auto res_tensor = builder.create<mlir::tosa::RFFT2dOp>(loc, output_real,
                                                           output_imag, input);

    // 此处报错TypeValue该构造函数不存在，所以不得已而注释
    // region.pool.addRankedTensor(
    //   TypeValue(output_real.dyn_cast<RankedTensorType>(), res_tensor),
    //   "tosa.rfft2d");
    return res_tensor.getOperation();
  };
}

// ::mlir::Type output, ::mlir::Value input1) {
// tosa.reciprocal (mlir::tosa::ReciprocalOp)
OpGenerator tosaReciprocalOpGenerator() {
  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto operandCandidates = region.pool.searchCandidatesFrom(
        {PoolType::StaticShapedTensor, PoolType::DynamicShapedTensor},
        emptyFilter());
    if (operandCandidates.empty()) {
      operandCandidates.push_back(region.pool.generateRankedTensor(
          builder, loc, randomRankedTensorType(builder.getContext())));
    }
    auto src1 = sampleTypedValueFrom(operandCandidates, "tosa.reciprocal");

    auto input1 = src1.val;
    auto output = src1.type;

    auto res = builder.create<mlir::tosa::ReciprocalOp>(loc, output, input1);
    region.pool.addRankedTensor(
        TypeValue(mlir::dyn_cast<RankedTensorType>(output), res), "tosa.reciprocal");
    return res.getOperation();
  };
}

//  ::mlir::Type output, ::mlir::Value input, uint64_t axis)
// tosa.reduce_all (mlir::tosa::ReduceAllOp)
OpGenerator tosaReduceAllOpGenerator() {
  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto operandCandidates = region.pool.searchCandidatesFrom(
        {PoolType::StaticShapedTensor, PoolType::DynamicShapedTensor},
        [](TypeValue tval) {
          return mlir::dyn_cast<RankedTensorType>(tval.type).getRank() > 0 &&
                 mlir::dyn_cast<RankedTensorType>(tval.type).getRank() <= 4;
        });
    if (operandCandidates.empty()) {
      SmallVector<int64_t> newShape;
      newShape.push_back(1);
      for (int i = 0; i < UR(3); i++) {
        newShape.push_back(1);
      }
      auto destTy = RankedTensorType::get(
          newShape, randomIntOrFloatType(builder.getContext()));
      auto dest = region.pool.generateRankedTensor(builder, loc, destTy);
      operandCandidates.push_back(dest);
    }
    auto src1 = sampleTypedValueFrom(operandCandidates, "tosa.reduce_all");

    auto input = src1.val;
    // auto output = src1.type;
    auto inputTy = mlir::dyn_cast<RankedTensorType>(src1.type);
    SmallVector<int64_t> ouputShape;
    ouputShape.push_back(1);
    for (int i = 1; i < inputTy.getShape().size(); i++) {
      ouputShape.push_back(inputTy.getDimSize(i));
    }
    auto ouputElementType = inputTy.getElementType();
    ouputShape[0] = 1;
    auto output = RankedTensorType::get(ouputShape, ouputElementType);

    mlir::IntegerAttr axis = builder.getI64IntegerAttr(0);
    // mlir::IntegerAttr axis = mlir::IntegerAttr::get(uint64_t(rand()),
    // builder.getContext());
    auto res =
        builder.create<mlir::tosa::ReduceAllOp>(loc, output, input, axis);
    region.pool.addRankedTensor(
        TypeValue(mlir::dyn_cast<RankedTensorType>(output), res), "tosa.reduce_all");
    return res.getOperation();
  };
}

// tosa.reduce_any (mlir::tosa::ReduceAnyOp)
// ::mlir::Type output, ::mlir::Value input, uint64_t axis) {
OpGenerator tosaReduceAnyOpGenerator() {
  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto operandCandidates = region.pool.searchCandidatesFrom(
        {PoolType::StaticShapedTensor, PoolType::DynamicShapedTensor},
        [](TypeValue tval) {
          return mlir::dyn_cast<RankedTensorType>(tval.type).getRank() > 0 &&
                 mlir::dyn_cast<RankedTensorType>(tval.type).getRank() <= 4;
        });
    if (operandCandidates.empty()) {
      SmallVector<int64_t> newShape;
      newShape.push_back(1);
      for (int i = 0; i < UR(3); i++) {
        newShape.push_back(1);
      }
      auto destTy = RankedTensorType::get(
          newShape, randomIntOrFloatType(builder.getContext()));
      auto dest = region.pool.generateRankedTensor(builder, loc, destTy);
      operandCandidates.push_back(dest);
    }
    auto src1 = sampleTypedValueFrom(operandCandidates, "tosa.reduce_any");

    auto input = src1.val;
    // auto output = src1.type;
    auto inputTy = mlir::dyn_cast<RankedTensorType>(src1.type);
    SmallVector<int64_t> ouputShape;
    ouputShape.push_back(1);
    for (int i = 1; i < inputTy.getShape().size(); i++) {
      ouputShape.push_back(inputTy.getDimSize(i));
    }
    auto ouputElementType = inputTy.getElementType();
    ouputShape[0] = 1;
    auto output = RankedTensorType::get(ouputShape, ouputElementType);
    mlir::IntegerAttr axis = builder.getI64IntegerAttr(0);
    // mlir::IntegerAttr axis = mlir::IntegerAttr::get(uint64_t(rand()),
    // builder.getContext());
    auto res =
        builder.create<mlir::tosa::ReduceAnyOp>(loc, output, input, axis);
    region.pool.addRankedTensor(
        TypeValue(mlir::dyn_cast<RankedTensorType>(output), res), "tosa.reduce_any");
    return res.getOperation();
  };
}

// tosa.reduce_max (mlir::tosa::ReduceMaxOp)
//  ::mlir::Type output, ::mlir::Value input, uint64_t axis) {
OpGenerator tosaReduceMaxOpGenerator() {
  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto operandCandidates = region.pool.searchCandidatesFrom(
        {PoolType::StaticShapedTensor, PoolType::DynamicShapedTensor},
        [](TypeValue tval) {
          return mlir::dyn_cast<RankedTensorType>(tval.type).getRank() > 0 &&
                 mlir::dyn_cast<RankedTensorType>(tval.type).getRank() <= 4;
        });
    if (operandCandidates.empty()) {
      SmallVector<int64_t> newShape;
      newShape.push_back(1);
      for (int i = 0; i < UR(3); i++) {
        newShape.push_back(1);
      }
      auto destTy = RankedTensorType::get(
          newShape, randomIntOrFloatType(builder.getContext()));
      auto dest = region.pool.generateRankedTensor(builder, loc, destTy);
      operandCandidates.push_back(dest);
    }
    auto src1 = sampleTypedValueFrom(operandCandidates, "tosa.reduce_max");

    auto input = src1.val;
    // auto output = src1.type;
    auto inputTy = mlir::dyn_cast<RankedTensorType>(src1.type);
    SmallVector<int64_t> ouputShape;
    ouputShape.push_back(1);
    for (int i = 1; i < inputTy.getShape().size(); i++) {
      ouputShape.push_back(inputTy.getDimSize(i));
    }
    auto ouputElementType = inputTy.getElementType();
    ouputShape[0] = 1;
    auto output = RankedTensorType::get(ouputShape, ouputElementType);
    mlir::IntegerAttr axis = builder.getI64IntegerAttr(0);
    // mlir::IntegerAttr axis = mlir::IntegerAttr::get(uint64_t(rand()),
    // builder.getContext());
    auto res =
        builder.create<mlir::tosa::ReduceMaxOp>(loc, output, input, axis);
    region.pool.addRankedTensor(
        TypeValue(mlir::dyn_cast<RankedTensorType>(output), res), "tosa.reduce_max");
    return res.getOperation();
  };
}

// tosa.reduce_min (mlir::tosa::ReduceMinOp)
// operand #0 must be unranked.tensor of number values or 1D/2D/3D/4D tensor of
// number values, but got 'tensor<f16>'
OpGenerator tosaReduceMinOpGenerator() {
  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto operandCandidates = region.pool.searchCandidatesFrom(
        {PoolType::StaticShapedTensor, PoolType::DynamicShapedTensor},
        [](TypeValue tval) {
          return mlir::dyn_cast<RankedTensorType>(tval.type).getRank() > 0 &&
                 mlir::dyn_cast<RankedTensorType>(tval.type).getRank() <= 4;
        });
    if (operandCandidates.empty()) {
      SmallVector<int64_t> newShape;
      newShape.push_back(1);
      for (int i = 0; i < UR(3); i++) {
        newShape.push_back(1);
      }
      auto destTy = RankedTensorType::get(
          newShape, randomIntOrFloatType(builder.getContext()));
      auto dest = region.pool.generateRankedTensor(builder, loc, destTy);
      operandCandidates.push_back(dest);
    }
    auto src1 = sampleTypedValueFrom(operandCandidates, "tosa.reduce_min");

    auto input = src1.val;
    // auto output = src1.type;
    auto inputTy = mlir::dyn_cast<RankedTensorType>(src1.type);
    SmallVector<int64_t> ouputShape;
    ouputShape.push_back(1);
    for (int i = 1; i < inputTy.getShape().size(); i++) {
      ouputShape.push_back(inputTy.getDimSize(i));
    }
    auto ouputElementType = inputTy.getElementType();
    ouputShape[0] = 1;
    auto output = RankedTensorType::get(ouputShape, ouputElementType);
    mlir::IntegerAttr axis = builder.getI64IntegerAttr(0);
    // mlir::IntegerAttr axis = mlir::IntegerAttr::get(uint64_t(rand()),
    // builder.getContext());
    auto res =
        builder.create<mlir::tosa::ReduceMinOp>(loc, output, input, axis);
    region.pool.addRankedTensor(
        TypeValue(mlir::dyn_cast<RankedTensorType>(output), res), "tosa.reduce_min");
    return res.getOperation();
  };
}
// tosa.reduce_prod (mlir::tosa::ReduceProdOp)
OpGenerator tosaReduceProdOpGenerator() {
  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto operandCandidates = region.pool.searchCandidatesFrom(
        {PoolType::StaticShapedTensor, PoolType::DynamicShapedTensor},
        [](TypeValue tval) {
          return mlir::dyn_cast<RankedTensorType>(tval.type).getRank() > 0 &&
                 mlir::dyn_cast<RankedTensorType>(tval.type).getRank() <= 4;
        });
    if (operandCandidates.empty()) {
      SmallVector<int64_t> newShape;
      newShape.push_back(1);
      for (int i = 0; i < UR(4); i++) {
        newShape.push_back(1);
      }
      auto destTy = RankedTensorType::get(
          newShape, randomIntOrFloatType(builder.getContext()));
      auto dest = region.pool.generateRankedTensor(builder, loc, destTy);
      operandCandidates.push_back(dest);
    }
    auto src1 = sampleTypedValueFrom(operandCandidates, "tosa.reduce_prod");

    auto input = src1.val;
    // auto output = src1.type;
    auto inputTy = mlir::dyn_cast<RankedTensorType>(src1.type);
    SmallVector<int64_t> ouputShape;
    ouputShape.push_back(1);
    for (int i = 1; i < inputTy.getShape().size(); i++) {
      ouputShape.push_back(inputTy.getDimSize(i));
    }
    auto ouputElementType = inputTy.getElementType();
    ouputShape[0] = 1;
    auto output = RankedTensorType::get(ouputShape, ouputElementType);
    mlir::IntegerAttr axis = builder.getI64IntegerAttr(0);
    // mlir::IntegerAttr axis = mlir::IntegerAttr::get(uint64_t(rand()),
    // builder.getContext());
    auto res =
        builder.create<mlir::tosa::ReduceProdOp>(loc, output, input, axis);
    region.pool.addRankedTensor(
        TypeValue(mlir::dyn_cast<RankedTensorType>(output), res),
        "tosa.reduce_prod");
    return res.getOperation();
  };
}
// tosa.reduce_sum (mlir::tosa::ReduceSumOp)
OpGenerator tosaReduceSumOpGenerator() {
  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto operandCandidates = region.pool.searchCandidatesFrom(
        {PoolType::StaticShapedTensor, PoolType::DynamicShapedTensor},
        [](TypeValue tval) {
          return mlir::dyn_cast<RankedTensorType>(tval.type).getRank() > 0 &&
                 mlir::dyn_cast<RankedTensorType>(tval.type).getRank() <= 4;
        });
    if (operandCandidates.empty()) {
      SmallVector<int64_t> newShape;
      newShape.push_back(1);
      for (int i = 0; i < UR(4); i++) {
        newShape.push_back(1);
      }
      auto destTy = RankedTensorType::get(
          newShape, randomIntOrFloatType(builder.getContext()));
      auto dest = region.pool.generateRankedTensor(builder, loc, destTy);
      operandCandidates.push_back(dest);
    }
    auto src1 = sampleTypedValueFrom(operandCandidates, "tosa.reduce_sum");

    auto input = src1.val;
    // auto output = src1.type;
    auto inputTy = mlir::dyn_cast<RankedTensorType>(src1.type);
    SmallVector<int64_t> ouputShape;
    ouputShape.push_back(1);
    for (int i = 1; i < inputTy.getShape().size(); i++) {
      ouputShape.push_back(inputTy.getDimSize(i));
    }
    auto ouputElementType = inputTy.getElementType();
    ouputShape[0] = 1;
    auto output = RankedTensorType::get(ouputShape, ouputElementType);

    mlir::IntegerAttr axis = builder.getI64IntegerAttr(0);
    // mlir::IntegerAttr axis = mlir::IntegerAttr::get(uint64_t(rand()),
    // builder.getContext());
    auto res =
        builder.create<mlir::tosa::ReduceSumOp>(loc, output, input, axis);
    region.pool.addRankedTensor(
        TypeValue(mlir::dyn_cast<RankedTensorType>(output), res), "tosa.reduce_sum");
    return res.getOperation();
  };
}

// tosa.rescale (mlir::tosa::RescaleOp)
// ::mlir::Type output, ::mlir::Value input,
//  ::mlir::IntegerAttr input_zp, ::mlir::IntegerAttr output_zp, :
//  ::mlir::DenseI32ArrayAttr multiplier, ::mlir::DenseI32ArrayAttr shift,
//  ::mlir::BoolAttr scale32, ::mlir::BoolAttr double_round, ::mlir::BoolAttr
//  per_channel
OpGenerator tosaRescaleOpGenerator() {
  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto operandCandidates = region.pool.searchCandidatesFrom(
        {PoolType::StaticShapedTensor, PoolType::DynamicShapedTensor},
        emptyFilter());
    if (operandCandidates.empty()) {
      operandCandidates.push_back(region.pool.generateRankedTensor(
          builder, loc, randomRankedTensorType(builder.getContext())));
    }
    auto src1 = sampleTypedValueFrom(operandCandidates, "tosa.rescale");
    auto input = src1.val;
    auto output = src1.type;

    auto input_zp = builder.getIntegerAttr(builder.getI32Type(),
                                           ((long long)rand() << 32) | rand());
    auto output_zp = builder.getIntegerAttr(builder.getI32Type(),
                                            ((long long)rand() << 32) | rand());

    SmallVector<int32_t> a;
    SmallVector<signed char> b;
    auto multiplier = builder.getDenseI32ArrayAttr(a);
    auto shift = builder.getDenseI8ArrayAttr(b);

    auto scale32 = builder.getBoolAttr(true);
    auto double_round = builder.getBoolAttr(true);
    auto per_channel = builder.getBoolAttr(true);

    auto res = builder.create<mlir::tosa::RescaleOp>(
        loc, output, input, input_zp, output_zp, multiplier, shift, scale32,
        double_round, per_channel);
    region.pool.addRankedTensor(
        TypeValue(mlir::dyn_cast<RankedTensorType>(output), res), "tosa.rescale");
    return res.getOperation();
  };
}

// tosa.reshape (mlir::tosa::ReshapeOp)
// ::mlir::Type output, ::mlir::Value input1, ::mlir::DenseI64ArrayAttr
// new_shape) {
OpGenerator tosaReshapeOpGenerator() {
  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto operandCandidates = region.pool.searchCandidatesFrom(
        {PoolType::StaticShapedTensor, PoolType::DynamicShapedTensor},
        emptyFilter());
    if (operandCandidates.empty()) {
      operandCandidates.push_back(region.pool.generateRankedTensor(
          builder, loc, randomRankedTensorType(builder.getContext())));
    }
    auto src1 = sampleTypedValueFrom(operandCandidates, "tosa.reshape");
    auto input1 = src1.val;
    auto output = src1.type;

    SmallVector<int64_t> a;
    auto new_shape = builder.getDenseI64ArrayAttr(a);

    auto res =
        builder.create<mlir::tosa::ReshapeOp>(loc, output, input1, new_shape);
    region.pool.addRankedTensor(
        TypeValue(mlir::dyn_cast<RankedTensorType>(output), res), "tosa.reshape");
    return res.getOperation();
  };
}

// tosa.resize (mlir::tosa::ResizeOp)
// ::mlir::Type output, ::mlir::Value input,
// ::mlir::DenseI64ArrayAttr scale, ::mlir::DenseI64ArrayAttr offset,
// ::mlir::DenseI64ArrayAttr border,
// ::mlir::StringAttr mode) {
OpGenerator tosaResizeOpGenerator() {
  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto operandCandidates = region.pool.searchCandidatesFrom(
        {PoolType::StaticShapedTensor, PoolType::DynamicShapedTensor},
        [](TypeValue tval) {
          return mlir::dyn_cast<RankedTensorType>(tval.type).getRank() == 4;
        });
    if (operandCandidates.empty()) {
      SmallVector<int64_t> newShape;
      for (int i = 0; i < 4; i++) {
        newShape.push_back(UR(100));
      }
      auto destTy = RankedTensorType::get(
          newShape, randomIntOrFloatType(builder.getContext()));
      auto dest = region.pool.generateRankedTensor(builder, loc, destTy);
      operandCandidates.push_back(dest);
    }
    auto src1 = sampleTypedValueFrom(operandCandidates, "tosa.resize");
    auto input = src1.val;
    auto output = src1.type;

    SmallVector<int64_t> a, b, c;
    for (int i = 0; i < 4; i++) {
      a.push_back(UR(100));
    }
    for (int i = 0; i < 2; i++) {
      b.push_back(UR(100));
    }
    for (int i = 0; i < 2; i++) {
      c.push_back(UR(100));
    }
    auto scale = builder.getDenseI64ArrayAttr(a);
    auto offset = builder.getDenseI64ArrayAttr(b);
    auto border = builder.getDenseI64ArrayAttr(c);
    // 'tosa.resize' op attribute 'mode' failed to satisfy constraint: Supported
    // resize/upsampling strategies

    // NearestNeighbor: 最近邻插值,简单快速但产生锯齿。
    // Linear: 双线性插值,较近邻产生更平滑结果但计算复杂度更高.
    // Area: 用平均值填充像素,保持像素总和不变但模糊原图。
    // Cubic: 三次样条插值,产生更光滑结果但计算开销最大。
    // Lanczos: 兰索斯窗口函数插值,在光滑度和计算效率上取得平衡。
    // MitchelNetravali: 一种二次Lanczos插值的变种算法。
    auto mode = builder.getStringAttr("Linear");

    auto res = builder.create<mlir::tosa::ResizeOp>(loc, output, input, scale,
                                                    offset, border, mode);
    region.pool.addRankedTensor(
        TypeValue(mlir::dyn_cast<RankedTensorType>(output), res), "tosa.resize");
    return res.getOperation();
  };
}

// tosa.reverse (mlir::tosa::ReverseOp)
// ::mlir::Type output, ::mlir::Value input, uint64_t axis) {
OpGenerator tosaReverseOpGenerator() {
  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto operandCandidates = region.pool.searchCandidatesFrom(
        {PoolType::StaticShapedTensor, PoolType::DynamicShapedTensor},
        [](TypeValue tval) {
          return mlir::dyn_cast<RankedTensorType>(tval.type).getRank() > 0 &&
                 mlir::dyn_cast<RankedTensorType>(tval.type).getRank() <= 4;
        });
    if (operandCandidates.empty()) {
      SmallVector<int64_t> newShape;
      newShape.push_back(1);
      for (int i = 0; i < UR(3); i++) {
        newShape.push_back(1);
      }

      auto destTy = RankedTensorType::get(
          newShape, randomIntOrFloatType(builder.getContext()));
      auto dest = region.pool.generateRankedTensor(builder, loc, destTy);
      operandCandidates.push_back(dest);
    }
    auto src1 = sampleTypedValueFrom(operandCandidates, "tosa.reverse");

    auto input = src1.val;

    auto inputTy = mlir::dyn_cast<RankedTensorType>(src1.type);

    SmallVector<int64_t> ouputShape;
    ouputShape.push_back(1);
    for (int i = 1; i < inputTy.getShape().size(); i++) {
      ouputShape.push_back(inputTy.getDimSize(i));
    }

    auto ouputElementType = inputTy.getElementType();
    ouputShape[0] = 1;
    auto output = RankedTensorType::get(ouputShape, ouputElementType);

    mlir::IntegerAttr axis = builder.getI64IntegerAttr(0);
    auto res = builder.create<mlir::tosa::ReverseOp>(loc, output, input, axis);
    region.pool.addRankedTensor(
        TypeValue(mlir::dyn_cast<RankedTensorType>(output), res), "tosa.reverse");
    return res.getOperation();
  };
}

// ::mlir::Type output, ::mlir::Value input1) {
// tosa.rsqrt (mlir::tosa::RsqrtOp)
OpGenerator tosaRsqrtOpGenerator() {
  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto operandCandidates = region.pool.searchCandidatesFrom(
        {PoolType::StaticShapedTensor, PoolType::DynamicShapedTensor},
        emptyFilter());
    if (operandCandidates.empty()) {
      operandCandidates.push_back(region.pool.generateRankedTensor(
          builder, loc, randomRankedTensorType(builder.getContext())));
    }
    auto src1 = sampleTypedValueFrom(operandCandidates, "tosa.rsqrt");

    auto input1 = src1.val;
    auto output = src1.type;

    auto res = builder.create<mlir::tosa::RsqrtOp>(loc, output, input1);
    region.pool.addRankedTensor(
        TypeValue(mlir::dyn_cast<RankedTensorType>(output), res), "tosa.rsqrt");
    return res.getOperation();
  };
}

// tosa.scatter (mlir::tosa::ScatterOp)
// ::mlir::Type values_out, ::mlir::Value values_in, ::mlir::Value indices,
// ::mlir::Value input) {
OpGenerator tosaScatterOpGenerator() {
  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    // unranked tensor of number values or 3D tensor of number values
    auto _3DCandidates = region.pool.searchCandidatesFrom(
        {PoolType::StaticShapedTensor, PoolType::DynamicShapedTensor},
        [](TypeValue tval) {
          return mlir::dyn_cast<RankedTensorType>(tval.type).getRank() == 3;
        });
    if (_3DCandidates.empty()) {
      SmallVector<int64_t> newShape;
      for (int i = 0; i < 3; i++) {
        newShape.push_back(1);
      }
      auto destTy = RankedTensorType::get(
          newShape, randomIntOrFloatType(builder.getContext()));
      auto dest = region.pool.generateRankedTensor(builder, loc, destTy);
      _3DCandidates.push_back(dest);
    }
    auto src0 = sampleTypedValueFrom(_3DCandidates, "tosa.scatter");
    auto src1 = region.pool.generateRankedTensor(
        builder, loc, mlir::dyn_cast<RankedTensorType>(src0.type));

    // 2D tensor of 32-bit signless integer values
    auto _2D32bitCandidates = region.pool.searchCandidatesFrom(
        {PoolType::StaticShapedTensor, PoolType::DynamicShapedTensor},
        [&](TypeValue tval) {
          return mlir::dyn_cast<RankedTensorType>(tval.type).getRank() == 2 &&
                 mlir::dyn_cast<RankedTensorType>(tval.type).getElementType() ==
                     builder.getI32Type();
        });
    if (_2D32bitCandidates.empty()) {
      SmallVector<int64_t> newShape;
      for (int i = 0; i < 2; i++) {
        newShape.push_back(1);
      }
      auto destTy = RankedTensorType::get(newShape, builder.getI32Type());
      auto dest = region.pool.generateRankedTensor(builder, loc, destTy);
      _2D32bitCandidates.push_back(dest);
    }
    auto src2 = sampleTypedValueFrom(_2D32bitCandidates, "tosa.scatter");

    auto values_out = src0.type;
    auto values_in = src0.val;
    auto indices = src2.val;
    auto input = src1.val;

    auto res = builder.create<mlir::tosa::ScatterOp>(loc, values_out, values_in,
                                                     indices, input);
    region.pool.addRankedTensor(
        TypeValue(mlir::dyn_cast<RankedTensorType>(values_out), res),
        "tosa.scatter");
    return res.getOperation();
  };
}

// tosa.select (mlir::tosa::SelectOp)
// ::mlir::Type output, ::mlir::Value pred, ::mlir::Value on_true, ::mlir::Value
// on_false) {
OpGenerator tosaSelectOpGenerator() {
  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto predCandidates = region.pool.searchCandidatesFrom(
        {PoolType::StaticShapedTensor, PoolType::DynamicShapedTensor},
        [&](TypeValue tVal) {
          return mlir::dyn_cast<RankedTensorType>(tVal.type).getElementType() ==
                 builder.getI1Type();
        });
    if (predCandidates.empty()) {
      SmallVector<int64_t> newShape;
      auto predTy = RankedTensorType::get(newShape, builder.getI1Type());
      auto dest = region.pool.generateRankedTensor(builder, loc, predTy);
      predCandidates.push_back(dest);
    }
    auto src0 = sampleTypedValueFrom(predCandidates, "tosa.select");

    // auto operandCandidates = region.pool.searchCandidatesFrom(
    //   {PoolType::StaticShapedTensor, PoolType::DynamicShapedTensor},
    //   emptyFilter());
    // if (operandCandidates.empty()) {
    //   operandCandidates.push_back(
    //     region.pool.generateRankedTensor(builder, loc,
    //     randomRankedTensorType(builder.getContext())));
    // }
    auto src1 = region.pool.generateRankedTensor(
        builder, loc, mlir::dyn_cast<RankedTensorType>(src0.type));
    auto src2 = region.pool.generateRankedTensor(
        builder, loc, mlir::dyn_cast<RankedTensorType>(src0.type));

    auto output = src0.type;
    auto pred = src0.val;
    auto on_true = src1.val;
    auto on_false = src2.val;

    auto res_tensor = builder.create<mlir::tosa::SelectOp>(loc, output, pred,
                                                           on_true, on_false);

    region.pool.addRankedTensor(
        TypeValue(mlir::dyn_cast<RankedTensorType>(output), res_tensor),
        "tosa.select");
    return res_tensor.getOperation();
  };
}

// tosa.sigmoid (mlir::tosa::SigmoidOp)
//  ::mlir::Type output, ::mlir::Value input) {
OpGenerator tosaSigmoidOpGenerator() {
  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto operandCandidates = region.pool.searchCandidatesFrom(
        {PoolType::StaticShapedTensor, PoolType::DynamicShapedTensor},
        emptyFilter());
    if (operandCandidates.empty()) {
      operandCandidates.push_back(region.pool.generateRankedTensor(
          builder, loc, randomRankedTensorType(builder.getContext())));
    }
    auto src1 = sampleTypedValueFrom(operandCandidates, "tosa.sigmoid");

    auto input = src1.val;
    auto output = src1.type;

    auto res = builder.create<mlir::tosa::SigmoidOp>(loc, output, input);
    region.pool.addRankedTensor(
        TypeValue(mlir::dyn_cast<RankedTensorType>(output), res), "tosa.sigmoid");
    return res.getOperation();
  };
}

// tosa.slice (mlir::tosa::SliceOp)
// ::mlir::Type output, ::mlir::Value input, ::mlir::DenseI64ArrayAttr start,
// ::mlir::DenseI64ArrayAttr size) {
OpGenerator tosaSliceOpGenerator() {
  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto operandCandidates = region.pool.searchCandidatesFrom(
        {PoolType::StaticShapedTensor, PoolType::DynamicShapedTensor},
        [](TypeValue tval) {
          return mlir::dyn_cast<RankedTensorType>(tval.type).getRank() > 0 &&
                 mlir::dyn_cast<RankedTensorType>(tval.type).getRank() <= 6;
        });
    if (operandCandidates.empty()) {
      operandCandidates.push_back(region.pool.generateRankedTensor(
          builder, loc, randomRankedTensorType(builder.getContext())));
    }
    auto src1 = sampleTypedValueFrom(operandCandidates, "tosa.slice");
    auto input = src1.val;
    auto output = src1.type;

    SmallVector<int64_t> a, b;
    auto start = builder.getDenseI64ArrayAttr(a);
    auto size = builder.getDenseI64ArrayAttr(b);

    auto res =
        builder.create<mlir::tosa::SliceOp>(loc, output, input, start, size);
    region.pool.addRankedTensor(
        TypeValue(mlir::dyn_cast<RankedTensorType>(output), res), "tosa.slice");
    return res.getOperation();
  };
}

// tosa.sub (mlir::tosa::SubOp)
// ::mlir::Type output, ::mlir::Value input1, ::mlir::Value input2) {
OpGenerator tosaSubOpGenerator() {
  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    // prepare operand
    auto operandCandidates = region.pool.searchCandidatesFrom(
        {PoolType::StaticShapedTensor, PoolType::DynamicShapedTensor},
        emptyFilter());
    if (operandCandidates.empty()) {
      operandCandidates.push_back(region.pool.generateRankedTensor(
          builder, loc, randomRankedTensorType(builder.getContext())));
    }
    auto src1 = sampleTypedValueFrom(operandCandidates, "tosa.sub");
    auto src2 = region.pool.generateRankedTensor(
        builder, loc, mlir::dyn_cast<RankedTensorType>(src1.type));

    auto input1 = src1.val;
    auto input2 = src2.val;
    auto output = src1.type;

    auto res_tensor =
        builder.create<mlir::tosa::SubOp>(loc, output, input1, input2);

    region.pool.addRankedTensor(
        TypeValue(mlir::dyn_cast<RankedTensorType>(output), res_tensor), "tosa.sub");
    return res_tensor.getOperation();
  };
}

// tosa.table (mlir::tosa::TableOp)
// ::mlir::Type output, ::mlir::Value input, ::mlir::Value table) {
OpGenerator tosaTableOpGenerator() {
  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto operandCandidates = region.pool.searchCandidatesFrom(
        {PoolType::StaticShapedTensor, PoolType::DynamicShapedTensor},
        emptyFilter());
    if (operandCandidates.empty()) {
      SmallVector<int64_t> newShape;
      newShape.push_back(1);
      for (int i = 0; i < UR(3); i++) {
        newShape.push_back(1);
      }
      auto destTy = RankedTensorType::get(
          newShape, randomIntOrFloatType(builder.getContext()));
      auto dest = region.pool.generateRankedTensor(builder, loc, destTy);
      operandCandidates.push_back(dest);
    }
    auto src1 = sampleTypedValueFrom(operandCandidates, "tosa.table");

    auto tableCandidates = region.pool.searchCandidatesFrom(
        {PoolType::StaticShapedTensor, PoolType::DynamicShapedTensor},
        [&](TypeValue tVal) {
          return mlir::dyn_cast<RankedTensorType>(tVal.type).getRank() == 1;
        });
    if (tableCandidates.empty()) {
      SmallVector<int64_t> newShape;
      newShape.push_back(1);
      auto tableTy = RankedTensorType::get(
          newShape, randomIntOrFloatType(builder.getContext()));
      auto dest = region.pool.generateRankedTensor(builder, loc, tableTy);
      tableCandidates.push_back(dest);
    }
    auto src2 = sampleTypedValueFrom(tableCandidates, "tosa.table");

    auto output = src1.type;
    auto input = src1.val;
    auto table = src2.val;

    auto res = builder.create<mlir::tosa::TableOp>(loc, output, input, table);
    region.pool.addRankedTensor(
        TypeValue(mlir::dyn_cast<RankedTensorType>(output), res), "tosa.table");
    return res.getOperation();
  };
}

// tosa.tanh (mlir::tosa::TanhOp)
OpGenerator tosaTanhOpGenerator() {
  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto operandCandidates = region.pool.searchCandidatesFrom(
        {PoolType::StaticShapedTensor, PoolType::DynamicShapedTensor},
        emptyFilter());
    if (operandCandidates.empty()) {
      operandCandidates.push_back(region.pool.generateRankedTensor(
          builder, loc, randomRankedTensorType(builder.getContext())));
    }
    auto src1 = sampleTypedValueFrom(operandCandidates, "tosa.tanh");

    auto input = src1.val;
    auto output = src1.type;

    auto res = builder.create<mlir::tosa::TanhOp>(loc, output, input);
    region.pool.addRankedTensor(
        TypeValue(mlir::dyn_cast<RankedTensorType>(output), res), "tosa.tanh");
    return res.getOperation();
  };
}

// tosa.tile (mlir::tosa::TileOp)
OpGenerator tosaTileOpGenerator() {
  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto operandCandidates = region.pool.searchCandidatesFrom(
        {PoolType::StaticShapedTensor, PoolType::DynamicShapedTensor},
        [](TypeValue tval) {
          return mlir::dyn_cast<RankedTensorType>(tval.type).getRank() > 0 &&
                 mlir::dyn_cast<RankedTensorType>(tval.type).getRank() <= 4;
        });
    if (operandCandidates.empty()) {
      operandCandidates.push_back(region.pool.generateRankedTensor(
          builder, loc, randomRankedTensorType(builder.getContext())));
    }
    auto src1 = sampleTypedValueFrom(operandCandidates, "tosa.tile");
    auto input = src1.val;
    auto output = src1.type;

    SmallVector<int64_t> a;
    auto multiples = builder.getDenseI64ArrayAttr(a);

    auto res =
        builder.create<mlir::tosa::TileOp>(loc, output, input, multiples);
    region.pool.addRankedTensor(
        TypeValue(mlir::dyn_cast<RankedTensorType>(output), res), "tosa.tile");
    return res.getOperation();
  };
}

// tosa.transpose_conv2d (mlir::tosa::TransposeConv2DOp)
// ::mlir::Type output, ::mlir::Value input, ::mlir::Value filter, ::mlir::Value
// bias,
// ::mlir::DenseI64ArrayAttr out_pad, ::mlir::DenseI64ArrayAttr stride,
// ::mlir::DenseI64ArrayAttr out_shape,
// /*optional*/mlir::tosa::ConvOpQuantizationAttr quantization_info) {
OpGenerator tosaTransposeConv2DOpGenerator() {
  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    auto inputCandidates = region.pool.searchCandidatesFrom(
        {PoolType::StaticShapedTensor, PoolType::DynamicShapedTensor},
        [](TypeValue tval) {
          return mlir::dyn_cast<RankedTensorType>(tval.type).getRank() == 4;
        });
    if (inputCandidates.empty()) {
      SmallVector<int64_t> newShape;
      for (int i = 0; i < 4; i++) {
        newShape.push_back(1);
      }
      auto destTy = RankedTensorType::get(
          newShape, randomIntOrFloatType(builder.getContext()));
      auto dest = region.pool.generateRankedTensor(builder, loc, destTy);
      inputCandidates.push_back(dest);
    }
    auto src1 = sampleTypedValueFrom(inputCandidates, "tosa.transpose_conv2d");
    auto input = src1.val;
    auto output = src1.type;

    auto biasCandidates = region.pool.searchCandidatesFrom(
        {PoolType::StaticShapedTensor, PoolType::DynamicShapedTensor},
        [](TypeValue tval) {
          return mlir::dyn_cast<RankedTensorType>(tval.type).getRank() == 1;
        });
    if (biasCandidates.empty()) {
      SmallVector<int64_t> newShape;
      newShape.push_back(1);
      auto destTy = RankedTensorType::get(
          newShape, randomIntOrFloatType(builder.getContext()));
      auto dest = region.pool.generateRankedTensor(builder, loc, destTy);
      biasCandidates.push_back(dest);
    }
    auto src2 = sampleTypedValueFrom(biasCandidates, "tosa.transpose_conv2d");
    auto bias = src2.val;

    SmallVector<Type> supported_types;
    // supported_types.push_back(builder.getI4Type());
    supported_types.push_back(builder.getI8Type());
    supported_types.push_back(builder.getF16Type());
    supported_types.push_back(builder.getF32Type());

    auto filterCandidates = region.pool.searchCandidatesFrom(
        {PoolType::StaticShapedTensor, PoolType::DynamicShapedTensor},
        [&](TypeValue tval) {
          return mlir::dyn_cast<RankedTensorType>(tval.type).getRank() == 4 &&
                 (
                     // mlir::dyn_cast<RankedTensorType>(tval.type).getElementType()
                     // == builder.getI4Type() ||
                     mlir::dyn_cast<RankedTensorType>(tval.type).getElementType() ==
                         builder.getI8Type() ||
                     mlir::dyn_cast<RankedTensorType>(tval.type).getElementType() ==
                         builder.getF32Type() ||
                     mlir::dyn_cast<RankedTensorType>(tval.type).getElementType() ==
                         builder.getF16Type());
        });
    if (filterCandidates.empty()) {
      SmallVector<int64_t> newShape;
      for (int i = 0; i < 4; i++) {
        newShape.push_back(UR(20));
      }
      auto destTy = RankedTensorType::get(newShape, supported_types[UR(3)]);
      auto dest = region.pool.generateRankedTensor(builder, loc, destTy);
      filterCandidates.push_back(dest);
    }
    auto src3 = sampleTypedValueFrom(filterCandidates, "tosa.transpose_conv2d");
    auto filter = src3.val;

    SmallVector<int64_t> a, b, c;
    for (int i = 0; i < 4; i++) {
      a.push_back(1);
    }
    for (int i = 0; i < 2; i++) {
      b.push_back(1);
    }
    for (int i = 0; i < 4; i++) {
      c.push_back(1);
    }
    auto out_pad = builder.getDenseI64ArrayAttr(a);
    auto stride = builder.getDenseI64ArrayAttr(b);
    auto out_shape = builder.getDenseI64ArrayAttr(c);

    auto res = builder.create<mlir::tosa::TransposeConv2DOp>(
        loc, output, input, filter, bias, out_pad, stride, out_shape);
    region.pool.addRankedTensor(
        TypeValue(mlir::dyn_cast<RankedTensorType>(output), res),
        "tosa.transpose_conv2d");
    return res.getOperation();
  };
}

// ::mlir::Type output, ::mlir::Value input1, ::mlir::Value perms) {
// tosa.transpose (mlir::tosa::TransposeOp)

OpGenerator tosaTransposeOpGenerator() {
  return [](OpBuilder &builder, Location loc, OpRegion &region) {
    // prepare operand
    auto operandCandidates = region.pool.searchCandidatesFrom(
        {PoolType::StaticShapedTensor, PoolType::DynamicShapedTensor},
        [](TypeValue tval) {
          return mlir::dyn_cast<RankedTensorType>(tval.type).getRank() > 0 &&
                 mlir::dyn_cast<RankedTensorType>(tval.type).getRank() <= 6;
        });
    if (operandCandidates.empty()) {
      operandCandidates.push_back(region.pool.generateRankedTensor(
          builder, loc, randomRankedTensorType(builder.getContext())));
    }
    auto src1 = sampleTypedValueFrom(operandCandidates, "tosa.transpose");

    auto permsCandidates = region.pool.searchCandidatesFrom(
        {PoolType::StaticShapedTensor, PoolType::DynamicShapedTensor},
        [&](TypeValue tval) {
          return mlir::dyn_cast<RankedTensorType>(tval.type).getElementType() ==
                     builder.getI32Type() ||
                 mlir::dyn_cast<RankedTensorType>(tval.type).getElementType() ==
                     builder.getI64Type();
        });
    if (permsCandidates.empty()) {
      SmallVector<int64_t> newShape;
      auto destTy = RankedTensorType::get(newShape, builder.getI64Type());
      if (UR(2) == 1) {
        auto destTy = RankedTensorType::get(newShape, builder.getI32Type());
      }
      auto dest = region.pool.generateRankedTensor(builder, loc, destTy);
      permsCandidates.push_back(dest);
    }
    auto src2 = sampleTypedValueFrom(permsCandidates, "tosa.transpose");

    auto output = src1.type;
    auto input1 = src1.val;
    auto perms = src2.val;

    auto res_tensor =
        builder.create<mlir::tosa::TransposeOp>(loc, output, input1, perms);

    region.pool.addRankedTensor(
        TypeValue(mlir::dyn_cast<RankedTensorType>(output), res_tensor),
        "tosa.transpose");
    return res_tensor.getOperation();
  };
}