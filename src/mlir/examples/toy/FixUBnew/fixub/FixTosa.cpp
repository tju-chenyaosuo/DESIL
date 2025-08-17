#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"

#include "fixub/DelOp.h"
#include "fixub/FixTosa.h"
#include "fixub/FixUBUtils.h"

void FixTosa::fixtosa(mlir::Operation *op) {
  std::string opname = op->getName().getStringRef().str();
  if (opname == "tosa.transpose_conv2d") {
    FixTosa::FixTranspose_conv2d(op);
  }
  return;
}

mlir::RankedTensorType
FixTosa::genStaticTensorTyFromExisting(mlir::ShapedType sty, mlir::Type elemTy,
                                       mlir::OpBuilder builder) {
  /*
   * Generate a new static shape from existing one,
   * the static dimension will be as same as sty,
   * and the dynamic dimension will be randomly generated.
   *
   * Params:
   * mlir::ShapedType sty, the existing shape, which may be a dynamic shaped
   * type. mlir::Type elemTy, the expected element type. mlir::OpBuilder
   * builder, to support OpBuilder.
   */
  llvm::SmallVector<int64_t> newShape = llvm::SmallVector<int64_t>();
  for (int64_t i = 0; i < sty.getRank(); i++) {
    if (sty.isDynamic(i) || sty.getDimSize(i) <= 0) {
      newShape.push_back(FixUBUtils::randsval(1, 32));
    } else {
      newShape.push_back(sty.getDimSize(i));
    }
  }
  return mlir::RankedTensorType::get(newShape, elemTy);
}

mlir::Type FixTosa::randomIntegerType(mlir::OpBuilder builder) {
  /*
   * Generate an random integer type.
   *
   * Params:
   * mlir::OpBuilder builder, to support MLIR-API calls.
   *
   * Return:
   * A random integer type.
   */
  mlir::Type randIntTy;
  switch (FixUBUtils::randsval(0, 5)) {
  case 0:
    randIntTy = builder.getI2Type();
    break;
  case 1:
    randIntTy = builder.getI4Type();
    break;
  case 2:
    randIntTy = builder.getI8Type();
    break;
  case 3:
    randIntTy = builder.getI16Type();
    break;
  case 4:
    randIntTy = builder.getI32Type();
    break;
  case 5:
    randIntTy = builder.getI64Type();
    break;
  }
  return randIntTy;
}

mlir::Type FixTosa::randomFloatType(mlir::OpBuilder builder) {
  /*
   * Generate an random float type.
   *
   * Params:
   * mlir::OpBuilder builder, to support MLIR-API calls.
   *
   * Return:
   * A random float type.
   */
  return FixUBUtils::randsval(0, 1) ? builder.getF16Type()
                                    : builder.getF32Type();
}

void FixTosa::FixTranspose_conv2d(mlir::Operation *op) {
  /*
   * Fix tosa.transpose_conv2d, the documentation is:
   * https://www.mlplatform.org/tosa/tosa_spec.html#_transpose_conv2d The
   * refined constraints are:
   *
   * 1. The result shape should be static.
   * 2. The result element type should not be i1.
   * 3. The input type should be static.
   * 4. When result element type is int/float, the input element type is
   *int/float.
   * 5. When result element type is float, the element type of bias and result
   *should be same.
   * 6. BC=OC=IC || BC=IC=1
   * 7. Corresponding shape should be same in the parameter.
   * 8. out_pad, out_shape, stride should be DenseI64ArrayAttr
   * 9. When result element type is integer, quantization_info should be
   *specified.
   * 10. ERROR_IF(in_t != i8_t  && input_zp != 0);
   * 11. ERROR_IF(weight_t != i8_t && weight_zp != 0);
   * 12. ERROR_IF(out_pad_top <= -KH || out_pad_bottom <= -KH);
   * 13. ERROR_IF(out_pad_left <= -KW || out_pad_right <= -KW);
   * 14. ERROR_IF(stride_y < 1 || stride_x < 1);
   * 15. ERROR_IF(OH != (IH - 1) * stride_y + out_pad_top + out_pad_bottom +
   *KH);
   * 16. ERROR_IF(OW != (IW - 1) * stride_x + out_pad_left + out_pad_right +
   *KW);
   *
   * Operation Example:
   * %108 = tosa.transpose_conv2d %arg0_input, %arg1_weights, %arg2_bias
   *				 {out_pad = array<i64: 0, 0, 0, 0>,
   * quantization_info = #tosa.conv_quant<input_zp = 0, weight_zp = 4200>,
   * out_shape = array<i64: 1, 32, 32, 16>, stride = array<i64: 1, 1>} :
   * (tensor<1x32x32x8xi16>, tensor<10x20x30x8xi8>, tensor<10xi16>) ->
   * tensor<1x51x16x10xi64>
   */

  // Set the environment.
  mlir::OpBuilder builder(op->getBlock(), op->getBlock()->begin());
  builder.setInsertionPoint(op);
  mlir::Location loc = op->getLoc();

  // Some condition variables
  bool genNewOne = false;
  bool genNewArg0 = false;
  bool genNewArg1 = false;
  bool genNewArg2 = false;

  // Read the corresponding infos.
  mlir::Value args0Input = op->getOperand(0);
  mlir::Value args1Weights = op->getOperand(1);
  mlir::Value args2Bias = op->getOperand(2);
  mlir::Value res = op->getResult(0);
  mlir::ShapedType args0InputSTy =
      args0Input.getType().dyn_cast<mlir::ShapedType>();
  mlir::ShapedType args1WeightsSTy =
      args1Weights.getType().dyn_cast<mlir::ShapedType>();
  mlir::ShapedType args2BiasSTy =
      args2Bias.getType().dyn_cast<mlir::ShapedType>();
  mlir::ShapedType resSTy = res.getType().dyn_cast<mlir::ShapedType>();

  // 1. The result shape should be static.
  if (genNewOne = FixUBUtils::isDynShape(resSTy)) {
    llvm::outs() << "[FixTosa::FixTranspose_conv2d] branch1\n";

    resSTy =
        genStaticTensorTyFromExisting(resSTy, resSTy.getElementType(), builder);
  }
  // 2. The result element type should not be i1.
  if (resSTy.getElementType() == builder.getI1Type()) {
    llvm::outs() << "[FixTosa::FixTranspose_conv2d] branch2\n";
    if (FixUBUtils::randsval(0, 1)) {
      llvm::outs() << "[FixTosa::FixTranspose_conv2d] branch3\n";
      resSTy = genStaticTensorTyFromExisting(resSTy, randomIntegerType(builder),
                                             builder);
    } else {
      llvm::outs() << "[FixTosa::FixTranspose_conv2d] branch4\n";
      resSTy = genStaticTensorTyFromExisting(resSTy, randomFloatType(builder),
                                             builder);
    }
    genNewOne = true;
  }

  // 3. The input type should be static.
  if (genNewArg0 = FixUBUtils::isDynShape(args0InputSTy)) {
    llvm::outs() << "[FixTosa::FixTranspose_conv2d] branch5\n";
    args0InputSTy = genStaticTensorTyFromExisting(
        args0InputSTy, args0InputSTy.getElementType(), builder);
  }
  if (genNewArg1 = FixUBUtils::isDynShape(args1WeightsSTy)) {
    llvm::outs() << "[FixTosa::FixTranspose_conv2d] branch6\n";
    args1WeightsSTy = genStaticTensorTyFromExisting(
        args1WeightsSTy, args1WeightsSTy.getElementType(), builder);
  }
  if (genNewArg2 = FixUBUtils::isDynShape(args2BiasSTy)) {
    llvm::outs() << "[FixTosa::FixTranspose_conv2d] branch7\n";
    args2BiasSTy = genStaticTensorTyFromExisting(
        args2BiasSTy, args2BiasSTy.getElementType(), builder);
  }

  // 4. When result element type is int/float, the input element type is
  // int/float.
  if (resSTy.getElementType() == builder.getF32Type() ||
      resSTy.getElementType() == builder.getF16Type()) {
    llvm::outs() << "[FixTosa::FixTranspose_conv2d] branch8\n";
    // 3.(1) input, weight, bias should have float element type.
    if (args0InputSTy.getElementType() != builder.getF32Type() ||
        args0InputSTy.getElementType() != builder.getF16Type()) {
      llvm::outs() << "[FixTosa::FixTranspose_conv2d] branch9\n";
      args0InputSTy = genStaticTensorTyFromExisting(
          args0InputSTy, randomIntegerType(builder), builder);
      genNewArg0 = true;
    }
    if (args1WeightsSTy.getElementType() != builder.getF32Type() ||
        args1WeightsSTy.getElementType() != builder.getF16Type()) {
      llvm::outs() << "[FixTosa::FixTranspose_conv2d] branch10\n";
      args1WeightsSTy = genStaticTensorTyFromExisting(
          args1WeightsSTy, randomIntegerType(builder), builder);
      genNewArg1 = true;
    }
    if (args2BiasSTy.getElementType() != builder.getF32Type() ||
        args2BiasSTy.getElementType() != builder.getF16Type()) {
      llvm::outs() << "[FixTosa::FixTranspose_conv2d] branch11\n";
      args2BiasSTy = genStaticTensorTyFromExisting(
          args2BiasSTy, randomIntegerType(builder), builder);
      genNewArg2 = true;
    }
    // 5. When result element type is float, the element type of bias and result
    // should be same.
    if (resSTy.getElementType() != args2BiasSTy.getElementType()) {
      llvm::outs() << "[FixTosa::FixTranspose_conv2d] branch12\n";
      args2BiasSTy = genStaticTensorTyFromExisting(
          args2BiasSTy, resSTy.getElementType(), builder);
      genNewArg2 = true;
    }
  }
  if (resSTy.getElementType().isIntOrIndex()) {
    llvm::outs() << "[FixTosa::FixTranspose_conv2d] branch13\n";
    args0InputSTy = genStaticTensorTyFromExisting(
        args0InputSTy, randomIntegerType(builder), builder);
    args1WeightsSTy = genStaticTensorTyFromExisting(
        args1WeightsSTy, randomIntegerType(builder), builder);
    args2BiasSTy = genStaticTensorTyFromExisting(
        args2BiasSTy, randomIntegerType(builder), builder);
  }

  int64_t resSTy_N = resSTy.getDimSize(0);
  int64_t resSTy_OH = resSTy.getDimSize(1);
  int64_t resSTy_OW = resSTy.getDimSize(2);
  int64_t resSTy_OC = resSTy.getDimSize(3);
  // ---
  int64_t args0InputSTy_N = args0InputSTy.getDimSize(0);
  int64_t args0InputSTy_IH = args0InputSTy.getDimSize(1);
  int64_t args0InputSTy_IW = args0InputSTy.getDimSize(2);
  int64_t args0InputSTy_IC = args0InputSTy.getDimSize(3);
  // ---
  int64_t args1WeightsSTy_OC = args1WeightsSTy.getDimSize(0);
  int64_t args1WeightsSTy_KH = args1WeightsSTy.getDimSize(1);
  int64_t args1WeightsSTy_KW = args1WeightsSTy.getDimSize(2);
  int64_t args1WeightsSTy_IC = args1WeightsSTy.getDimSize(3);
  // ---
  int64_t args2BiasSTy_BC = args2BiasSTy.getDimSize(0);

  int64_t const_N = resSTy_N;
  int64_t const_OH = resSTy_OH;
  int64_t const_OW = resSTy_OW;
  int64_t const_OC = resSTy_OC;
  int64_t const_IH = args0InputSTy_IH;
  int64_t const_IW = args0InputSTy_IW;
  int64_t const_IC = args0InputSTy_IC;
  int64_t const_KH = args1WeightsSTy_KH;
  int64_t const_KW = args1WeightsSTy_KW;
  int64_t const_BC = args2BiasSTy_BC;

  // 6. BC=OC=IC || BC=IC=1
  if (!((const_BC == const_OC && const_BC == const_IC) ||
        (const_BC == 1 && const_IC == const_BC))) {
    llvm::outs() << "[FixTosa::FixTranspose_conv2d] branch14\n";
    if (FixUBUtils::randsval(0, 1)) {
      llvm::outs() << "[FixTosa::FixTranspose_conv2d] branch15\n";
      const_BC = const_OC;
      const_IC = const_OC;
    } else {
      llvm::outs() << "[FixTosa::FixTranspose_conv2d] branch16\n";
      const_BC = 1;
      const_IC = 1;
    }
  }

  // 7. Corresponding shape should be same in the parameter.
  llvm::SmallVector<int64_t> newArg0InputShape = llvm::SmallVector<int64_t>();
  newArg0InputShape.push_back(args0InputSTy_N);
  newArg0InputShape.push_back(args0InputSTy_IH);
  newArg0InputShape.push_back(args0InputSTy_IW);
  newArg0InputShape.push_back(args0InputSTy_IC);
  llvm::SmallVector<int64_t> newArg1WeightsShape = llvm::SmallVector<int64_t>();
  newArg1WeightsShape.push_back(args1WeightsSTy_OC);
  newArg1WeightsShape.push_back(args1WeightsSTy_KH);
  newArg1WeightsShape.push_back(args1WeightsSTy_KW);
  newArg1WeightsShape.push_back(args1WeightsSTy_IC);
  llvm::SmallVector<int64_t> newArg2BiasShape = llvm::SmallVector<int64_t>();
  newArg2BiasShape.push_back(args2BiasSTy_BC);

  if (args0InputSTy_N != const_N || args0InputSTy_IH != const_IH ||
      args0InputSTy_IW != const_IW || args0InputSTy_IC != const_IC) {
    llvm::outs() << "[FixTosa::FixTranspose_conv2d] branch17\n";
    newArg0InputShape[0] = const_N;
    newArg0InputShape[1] = const_IH;
    newArg0InputShape[2] = const_IW;
    newArg0InputShape[3] = const_IC;
    genNewArg0 = true;
  }
  if (args1WeightsSTy_OC != const_OC || args1WeightsSTy_KH != const_KH ||
      args1WeightsSTy_KW != const_KW || args1WeightsSTy_IC != const_IC) {
    llvm::outs() << "[FixTosa::FixTranspose_conv2d] branch18\n";
    newArg1WeightsShape[0] = const_OC;
    newArg1WeightsShape[1] = const_KH;
    newArg1WeightsShape[2] = const_KW;
    newArg1WeightsShape[3] = const_IC;
    genNewArg1 = true;
  }
  if (args2BiasSTy_BC != const_BC) {
    llvm::outs() << "[FixTosa::FixTranspose_conv2d] branch19\n";
    newArg2BiasShape[0] = const_BC;
    genNewArg2 = true;
  }

  // reload
  resSTy_N = resSTy.getDimSize(0);
  resSTy_OH = resSTy.getDimSize(1);
  resSTy_OW = resSTy.getDimSize(2);
  resSTy_OC = resSTy.getDimSize(3);
  // ---
  args2BiasSTy_BC = args2BiasSTy.getDimSize(0);
  // ---
  args1WeightsSTy_OC = args1WeightsSTy.getDimSize(0);
  args1WeightsSTy_KH = args1WeightsSTy.getDimSize(1);
  args1WeightsSTy_KW = args1WeightsSTy.getDimSize(2);
  args1WeightsSTy_IC = args1WeightsSTy.getDimSize(3);
  // ---
  args0InputSTy_N = args0InputSTy.getDimSize(0);
  args0InputSTy_IH = args0InputSTy.getDimSize(1);
  args0InputSTy_IW = args0InputSTy.getDimSize(2);
  args0InputSTy_IC = args0InputSTy.getDimSize(3);

  bool newOutPadAttr = false;
  bool newOutShapeAttr = false;
  bool newStrideAttr = false;
  bool newQuantizationInfoAttr = false;

  // 8. out_pad, out_shape, stride should be DenseI64ArrayAttr
  mlir::DenseI64ArrayAttr outPadAttr =
      op->getAttr("out_pad").dyn_cast<mlir::DenseI64ArrayAttr>();
  mlir::DenseI64ArrayAttr outShapeAttr =
      op->getAttr("out_shape").dyn_cast<mlir::DenseI64ArrayAttr>();
  mlir::DenseI64ArrayAttr strideAttr =
      op->getAttr("stride").dyn_cast<mlir::DenseI64ArrayAttr>();
  mlir::tosa::ConvOpQuantizationAttr quantizationInfoAttr =
      op->getAttr("quantization_info")
          .dyn_cast<mlir::tosa::ConvOpQuantizationAttr>();
  if (!outPadAttr) {
    llvm::outs() << "[FixTosa::FixTranspose_conv2d] branch20\n";
    llvm::SmallVector<int64_t> outPadAttrArray = llvm::SmallVector<int64_t>();
    outPadAttrArray.push_back(FixUBUtils::randsval(1, 10));
    outPadAttrArray.push_back(FixUBUtils::randsval(1, 10));
    outPadAttrArray.push_back(FixUBUtils::randsval(1, 10));
    outPadAttr = builder.getDenseI64ArrayAttr(outPadAttrArray);
    newOutPadAttr = true;
  }
  if (!outShapeAttr) {
    llvm::outs() << "[FixTosa::FixTranspose_conv2d] branch21\n";
    llvm::SmallVector<int64_t> outShapeAttrArray = llvm::SmallVector<int64_t>();
    outShapeAttrArray.push_back(resSTy_N);
    outShapeAttrArray.push_back(resSTy_OH);
    outShapeAttrArray.push_back(resSTy_OW);
    outShapeAttrArray.push_back(resSTy_OC);
    outShapeAttr = builder.getDenseI64ArrayAttr(outShapeAttrArray);
    newOutShapeAttr = true;
  }
  if (genNewOne = !strideAttr) {
    llvm::outs() << "[FixTosa::FixTranspose_conv2d] branch22\n";
    llvm::SmallVector<int64_t> strideAttrArray = llvm::SmallVector<int64_t>();
    strideAttrArray.push_back(FixUBUtils::randsval(1, 5));
    strideAttrArray.push_back(FixUBUtils::randsval(1, 5));
    strideAttr = builder.getDenseI64ArrayAttr(strideAttrArray);
    newStrideAttr = true;
  }

  int64_t inputZp;
  int64_t weightZp;

  // 9. When result element type is integer, quantization_info should be
  // specified.
  if (resSTy.getElementType().isIntOrIndex() && !quantizationInfoAttr) {
    llvm::outs() << "[FixTosa::FixTranspose_conv2d] branch23\n";
    inputZp = FixUBUtils::randsval(64);
    weightZp = FixUBUtils::randsval(64);
    quantizationInfoAttr =
        builder.getAttr<mlir::tosa::ConvOpQuantizationAttr>(inputZp, weightZp);
    newQuantizationInfoAttr = true;
  }

  if (quantizationInfoAttr) {
    llvm::outs() << "[FixTosa::FixTranspose_conv2d] branch24\n";
    inputZp = quantizationInfoAttr.getInputZp();
    weightZp = quantizationInfoAttr.getWeightZp();
  }

  // 10. ERROR_IF(in_t != i8_t  && input_zp != 0);
  if (args0InputSTy.getElementType() != builder.getI8Type() &&
      quantizationInfoAttr &&
      inputZp != 0 /* because &&, here is not an ub */) {
    llvm::outs() << "[FixTosa::FixTranspose_conv2d] branch25\n";
    inputZp = 0;
    newQuantizationInfoAttr = true;
  }

  // 11. ERROR_IF(weight_t != i8_t && weight_zp != 0);
  if (args1WeightsSTy.getElementType() != builder.getI8Type() &&
      quantizationInfoAttr && weightZp != 0) {
    llvm::outs() << "[FixTosa::FixTranspose_conv2d] branch26\n";
    weightZp = 0;
    newQuantizationInfoAttr = true;
  }
  if (newQuantizationInfoAttr) {
    llvm::outs() << "[FixTosa::FixTranspose_conv2d] branch27\n";
    quantizationInfoAttr =
        builder.getAttr<mlir::tosa::ConvOpQuantizationAttr>(inputZp, weightZp);
  }

  auto outPadAttrArray = outPadAttr.asArrayRef();
  int64_t out_pad_top = outPadAttrArray[0];
  int64_t out_pad_bottom = outPadAttrArray[1];
  int64_t out_pad_left = outPadAttrArray[2];
  int64_t out_pad_right = outPadAttrArray[3];

  llvm::SmallVector<int64_t> newOutPadAttrArray = llvm::SmallVector<int64_t>();
  newOutPadAttrArray.push_back(out_pad_top);
  newOutPadAttrArray.push_back(out_pad_bottom);
  newOutPadAttrArray.push_back(out_pad_left);
  newOutPadAttrArray.push_back(out_pad_right);
  // 12. ERROR_IF(out_pad_top <= -KH || out_pad_bottom <= -KH);
  // 13. ERROR_IF(out_pad_left <= -KW || out_pad_right <= -KW);
  if (out_pad_top <= -args1WeightsSTy_KH) {
    llvm::outs() << "[FixTosa::FixTranspose_conv2d] branch28\n";
    newOutPadAttrArray[0] =
        FixUBUtils::randsval(-args1WeightsSTy_KH + 1, args1WeightsSTy_KH * 2);
    newOutPadAttr = true;
  }
  if (out_pad_bottom <= -args1WeightsSTy_KH) {
    llvm::outs() << "[FixTosa::FixTranspose_conv2d] branch29\n";
    newOutPadAttrArray[1] =
        FixUBUtils::randsval(-args1WeightsSTy_KH + 1, args1WeightsSTy_KH * 2);
    newOutPadAttr = true;
  }
  if (out_pad_left <= -args1WeightsSTy_KW) {
    llvm::outs() << "[FixTosa::FixTranspose_conv2d] branch30\n";
    newOutPadAttrArray[2] =
        FixUBUtils::randsval(-args1WeightsSTy_KW + 1, args1WeightsSTy_KW * 2);
    newOutPadAttr = true;
  }
  if (out_pad_right <= -args1WeightsSTy_KW) {
    llvm::outs() << "[FixTosa::FixTranspose_conv2d] branch31\n";
    newOutPadAttrArray[3] =
        FixUBUtils::randsval(-args1WeightsSTy_KW + 1, args1WeightsSTy_KW * 2);
    newOutPadAttr = true;
  }
  if (newOutPadAttr) {
    llvm::outs() << "[FixTosa::FixTranspose_conv2d] branch32\n";
    outPadAttr = builder.getDenseI64ArrayAttr(newOutPadAttrArray);
  }

  // 14. ERROR_IF(stride_y < 1 || stride_x < 1);
  auto strideAttrArray = strideAttr.asArrayRef();
  int64_t stride_y = strideAttrArray[0];
  int64_t stride_x = strideAttrArray[1];

  llvm::SmallVector<int64_t> newStrideAttrArray = llvm::SmallVector<int64_t>();
  newStrideAttrArray.push_back(stride_y);
  newStrideAttrArray.push_back(stride_x);
  if (stride_x < 1) {
    llvm::outs() << "[FixTosa::FixTranspose_conv2d] branch33\n";
    newStrideAttrArray[1] = FixUBUtils::randsval(1, 5);
    newStrideAttr = true;
  }
  if (stride_y < 1) {
    llvm::outs() << "[FixTosa::FixTranspose_conv2d] branch34\n";
    newStrideAttrArray[0] = FixUBUtils::randsval(1, 5);
    newStrideAttr = true;
  }
  if (newStrideAttr) {
    llvm::outs() << "[FixTosa::FixTranspose_conv2d] branch35\n";
    strideAttr = builder.getDenseI64ArrayAttr(newStrideAttrArray);
  }

  // 15. ERROR_IF(OH != (IH - 1) * stride_y + out_pad_top + out_pad_bottom +
  // KH);
  // 16. ERROR_IF(OW != (IW - 1) * stride_x + out_pad_left + out_pad_right +
  // KW);
  strideAttrArray = strideAttr.asArrayRef();
  stride_y = strideAttrArray[0];
  stride_x = strideAttrArray[1];
  outPadAttrArray = outPadAttr.asArrayRef();
  out_pad_top = outPadAttrArray[0];
  out_pad_bottom = outPadAttrArray[1];
  out_pad_left = outPadAttrArray[2];
  out_pad_right = outPadAttrArray[3];
  if (resSTy_OH != (args0InputSTy_IH - 1) * stride_y + out_pad_top +
                       out_pad_bottom + args1WeightsSTy_KH) {
    llvm::outs() << "[FixTosa::FixTranspose_conv2d] branch36\n";
    resSTy_OH = (args0InputSTy_IH - 1) * stride_y + out_pad_top +
                out_pad_bottom + args1WeightsSTy_KH;
    genNewOne = true;
    newOutShapeAttr = true;
  }
  if (resSTy_OW != (args0InputSTy_IW - 1) * stride_x + out_pad_left +
                       out_pad_right + args1WeightsSTy_KW) {
    llvm::outs() << "[FixTosa::FixTranspose_conv2d] branch37\n";
    resSTy_OW = (args0InputSTy_IW - 1) * stride_x + out_pad_left +
                out_pad_right + args1WeightsSTy_KW;
    genNewOne = true;
    newOutShapeAttr = true;
  }
  if (genNewOne) {
    llvm::outs() << "[FixTosa::FixTranspose_conv2d] branch38\n";
    llvm::SmallVector<int64_t> newResShape = llvm::SmallVector<int64_t>();
    newResShape.push_back(resSTy_N);
    newResShape.push_back(resSTy_OH);
    newResShape.push_back(resSTy_OW);
    newResShape.push_back(resSTy_OC);
    resSTy =
        genStaticTensorTyFromExisting(resSTy, resSTy.getElementType(), builder);
  }
  if (newOutShapeAttr) {
    llvm::outs() << "[FixTosa::FixTranspose_conv2d] branch39\n";
    llvm::SmallVector<int64_t> outShapeAttrArray = llvm::SmallVector<int64_t>();
    outShapeAttrArray.push_back(resSTy_N);
    outShapeAttrArray.push_back(resSTy_OH);
    outShapeAttrArray.push_back(resSTy_OW);
    outShapeAttrArray.push_back(resSTy_OC);
    outShapeAttr = builder.getDenseI64ArrayAttr(outShapeAttrArray);
  }

  // generate a new one or repalce some operand.
  if (genNewOne) {
    llvm::outs() << "[FixTosa::FixTranspose_conv2d] branch40\n";
    args0Input =
        FixUBUtils::genRandomTensorFromExisting(args0InputSTy, builder, loc);
    args1Weights =
        FixUBUtils::genRandomTensorFromExisting(args1WeightsSTy, builder, loc);
    args2Bias =
        FixUBUtils::genRandomTensorFromExisting(args2BiasSTy, builder, loc);
    if (newQuantizationInfoAttr) {
      llvm::outs() << "[FixTosa::FixTranspose_conv2d] branch41\n";
      mlir::Value res =
          builder
              .create<mlir::tosa::TransposeConv2DOp>(
                  loc, resSTy, args0Input, args1Weights, args2Bias, outPadAttr,
                  strideAttr, outShapeAttr, quantizationInfoAttr)
              .getOperation()
              ->getResult(0);
      mlir::ShapedType oriResSTy =
          op->getResult(0).getType().dyn_cast<mlir::ShapedType>();
      mlir::Value newVal =
          FixUBUtils::genRandomTensorFromExisting(oriResSTy, builder, loc);
      op->replaceAllUsesWith(newVal.getDefiningOp());
      delOps.push_back(op);
    } else {
      llvm::outs() << "[FixTosa::FixTranspose_conv2d] branch42\n";
      if (genNewArg0) {
        llvm::outs() << "[FixTosa::FixTranspose_conv2d] branch43\n";
        op->setOperand(0, args0Input);
      }
      if (genNewArg1) {
        llvm::outs() << "[FixTosa::FixTranspose_conv2d] branch44\n";
        op->setOperand(1, args1Weights);
      }
      if (genNewArg2) {
        llvm::outs() << "[FixTosa::FixTranspose_conv2d] branch45\n";
        op->setOperand(2, args2Bias);
      }
      if (newOutPadAttr) {
        llvm::outs() << "[FixTosa::FixTranspose_conv2d] branch46\n";
        op->setAttr("out_pad", outPadAttr);
      }
      if (newOutShapeAttr) {
        llvm::outs() << "[FixTosa::FixTranspose_conv2d] branch47\n";
        op->setAttr("out_shape", outShapeAttr);
      }
      if (newQuantizationInfoAttr) {
        llvm::outs() << "[FixTosa::FixTranspose_conv2d] branch48\n";
        op->setAttr("quantization_info", quantizationInfoAttr);
      }
      if (newStrideAttr) {
        llvm::outs() << "[FixTosa::FixTranspose_conv2d] branch49\n";
        op->setAttr("stride", strideAttr);
      }
    }
  }
  return;
}
