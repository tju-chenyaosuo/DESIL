//
// Created by Stan Wang on 2022/9/13.
//
#ifndef OP_GENERATOR_PROVIDER_H
#define OP_GENERATOR_PROVIDER_H

#include "../TypedValuePool.h"
#include "../Util.h"
#include <map>
#include <set>

// using namespace mlir;

inline std::vector<mlir::arith::AtomicRMWKind> floatRmwKinds = {
    mlir::arith::AtomicRMWKind::addf, mlir::arith::AtomicRMWKind::assign,
    // mlir::arith::AtomicRMWKind::maxf, mlir::arith::AtomicRMWKind::minf,
    mlir::arith::AtomicRMWKind::mulf};

inline std::vector<mlir::arith::AtomicRMWKind> intRmwKinds = {
    mlir::arith::AtomicRMWKind::addi, mlir::arith::AtomicRMWKind::assign,
    mlir::arith::AtomicRMWKind::maxs, mlir::arith::AtomicRMWKind::maxu,
    mlir::arith::AtomicRMWKind::mins, mlir::arith::AtomicRMWKind::minu,
    mlir::arith::AtomicRMWKind::muli, mlir::arith::AtomicRMWKind::ori,
    mlir::arith::AtomicRMWKind::andi};

// TODO define parent region - define struct recursively
struct OpRegion {
  std::string parent_op;
  int depth;
  std::vector<std::string> filter;
  std::vector<std::string> terminatorOp;
  TypedValuePool pool;

  OpRegion(std::string op, int depth) : parent_op(op), depth(depth) {
    pool = TypedValuePool();
    filter = std::vector<std::string>();
    terminatorOp = std::vector<std::string>();
  }
};

using OpGenerator =
    std::function<mlir::Operation *(mlir::OpBuilder &, mlir::Location, OpRegion &)>;

// TODO- some terminators are optional
// for example, scf.yield
inline std::map<std::string, std::vector<std::string>> terminatorMap = {
    {"func.return", {"func.func"}},
    {"linalg.yield", {"linalg.generic", "linalg.map", "linalg.reduce"}}};

OpGenerator emptyGenerator();

struct OpGen {
  std::string opName;
  // We do not maintain these flags for each Op Generation since the requirement
  // ] are specified by only a little ops. bool hasSideEffect;
  OpGenerator generator;

  OpGen(std::string opName = "empty", OpGenerator generator = emptyGenerator())
      : opName(opName), generator(generator) {}

  mlir::Operation *apply(mlir::OpBuilder &builder, mlir::Location loc, OpRegion &region) {
    auto op = generator(builder, loc, region);
    return op;
  }

  bool isTerminatorOf(OpRegion opRegion) {
    return terminatorMap.find(opName) != terminatorMap.end() &&
           std::find(terminatorMap[opName].begin(), terminatorMap[opName].end(),
                     opRegion.parent_op) != terminatorMap[opName].end();
  }
};

// global(module) function pool
inline std::vector<mlir::func::FuncOp> funcPool;

OpGenerator initGenerator();

/*-------------- bufferization generators -------------*/
/*
OpGenerator bufferizationAllocTensorGenerator();
OpGenerator bufferizationCloneGenerator();
OpGenerator bufferizationDeallocGenerator();
OpGenerator bufferizationToMemrefGenerator();
OpGenerator bufferizationToTensorGenerator();
*/
/*---------------- linalg operation generators ---------------*/
OpGenerator linalgGenericGenerator();
// OpGenerator randomLinalgElementwiseGenerator();
OpGenerator linalgMapGenerator();
OpGenerator linalgReduceGenerator();
OpGenerator linalgMatMulGenerator();
OpGenerator linalgTransposeGenerator();
OpGenerator linalgBroadCastGenerator();
OpGenerator linalgCopyGenerator();
OpGenerator linalgVecmatGenerator();
OpGenerator linalgDotGenerator();
OpGenerator linalgBatchMatmulGenerator();
OpGenerator linalgBatchMatmulTransposeBGenerator();
OpGenerator linalgBatchMatvecGenerator();
OpGenerator linalgBatchReduceMatmulGenerator();
OpGenerator linalgConv1dNcwFcwGenerator();
OpGenerator linalgConv1dNwcWcfGenerator();
OpGenerator linalgConv1dGenerator();
OpGenerator linalgConv2dNchwFchwGenerator();
OpGenerator linalgConv2dNgchwFgchwGenerator();
OpGenerator linalgConv2dBhwcFhwcGenerator();
OpGenerator linalgConv2dBhwcHwcfGenerator();
OpGenerator linalgConv2dBhwcHwcfQGenerator();
OpGenerator linalgConv2dGenerator();
OpGenerator linalgConv3dNdhwcDhwcfGenerator();
OpGenerator linalgConv3dNdhwcDhwcfQGenerator();
OpGenerator linalgConv3dGenerator();
OpGenerator linalgDepthwiseConv1dNwcWcGenerator();
OpGenerator linalgDepthwiseConv1dNwcWcmGenerator();
OpGenerator linalgDepthwiseConv2dNchwChwGenerator();
OpGenerator linalgDepthwiseConv2dNhwcHwcGenerator();
OpGenerator linalgDepthwiseConv2dNhwcHwcQGenerator();
OpGenerator linalgDepthwiseConv2dNhwcHwcmGenerator();
OpGenerator linalgDepthwiseConv2dNhwcHwcmQGenerator();
OpGenerator linalgDepthwiseConv3dNdhwcDhwcGenerator();
OpGenerator linalgDepthwiseConv3dNdhwcDhwcmGenerator();
OpGenerator linalgElemwiseBinaryGenerator();
OpGenerator linalgElemwiseUnaryGenerator();
OpGenerator linalgFillGenerator();
OpGenerator linalgFillRng2dGenerator();
OpGenerator linalgIndexGenerator();
OpGenerator linalgMatmulTransposeBGenerator();
OpGenerator linalgMatmulUnsignedGenerator();
OpGenerator linalgMatvecGenerator();
OpGenerator linalgMmt4dGenerator();
OpGenerator linalgPoolingNchwMaxGenerator();
OpGenerator linalgPoolingNchwSumGenerator();
OpGenerator linalgPoolingNcwMaxGenerator();
OpGenerator linalgPoolingNcwSumGenerator();
OpGenerator linalgPoolingNdhwcMaxGenerator();
OpGenerator linalgPoolingNdhwcMinGenerator();
OpGenerator linalgPoolingNdhwcSumGenerator();
OpGenerator linalgPoolingNhwcMaxGenerator();
OpGenerator linalgPoolingNhwcMaxUnsignedGenerator();
OpGenerator linalgPoolingNhwcMinGenerator();
OpGenerator linalgPoolingNhwcMinUnsignedGenerator();
OpGenerator linalgPoolingNhwcSumGenerator();
OpGenerator linalgPoolingNwcMaxGenerator();
OpGenerator linalgPoolingNwcMaxUnsignedGenerator();
OpGenerator linalgPoolingNwcMinGenerator();
OpGenerator linalgPoolingNwcMinUnsignedGenerator();
OpGenerator linalgPoolingNwcSumGenerator();
OpGenerator linalgQuantizedBatchMatmulGenerator();
OpGenerator linalgQuantizedMatmulGenerator();

/*---------------- memref operation generators -------------------*/
OpGenerator assumeAlignmentGenerator();
OpGenerator reallocGenerator();
OpGenerator allocGenerator();
// OpGenerator tensorStoreGenerator();
OpGenerator memrefLoadGenerator();
OpGenerator memrefStoreGenerator();
// OpGenerator atomicRMWGenerator();
// OpGenerator genericAtomicRMWGenerator();
OpGenerator memrefCopyGenerator();
OpGenerator allocaGenerator();
// OpGenerator allocaScopeGenerator();
OpGenerator memrefCastGenerator();
// OpGenerator subViewGenerator();
// OpGenerator transposeGenerator();

/*---------------- arith operation generators -------------------*/
OpGenerator addFGenerator();
OpGenerator addIGenerator();
OpGenerator andIGenerator();
OpGenerator ceilDivSIGenerator();
OpGenerator ceilDivUIGenerator();
OpGenerator cmpFGenerator();
OpGenerator cmpIGenerator();
OpGenerator constantGenerator();
OpGenerator divFGenerator();
OpGenerator divSIGenerator();
OpGenerator divUIGenerator();
OpGenerator floorDivSIGenerator();
OpGenerator maxFGenerator();
OpGenerator maxUIGenerator();
OpGenerator maxSIGenerator();
OpGenerator minFGenerator();
OpGenerator minUIGenerator();
OpGenerator minSIGenerator();
OpGenerator mulFGenerator();
OpGenerator mulIGenerator();
OpGenerator negFGenerator();
OpGenerator orIGenerator();
OpGenerator remFGenerator();
OpGenerator remSIGenerator();
OpGenerator remUIGenerator();
OpGenerator shlIGenerator();
OpGenerator shrSIGenerator();
OpGenerator shrUIGenerator();
OpGenerator subFGenerator();
OpGenerator subIGenerator();
OpGenerator xorIGenerator();

/*---------------- math operation generators -------------------*/
OpGenerator absFGenerator();     // f u
OpGenerator absIGenerator();     // f u
OpGenerator absIGeimanerator();     // i u
OpGenerator atanGenerator();     // float unary
OpGenerator atan2Generator();    // float binary
OpGenerator ceilGenerator();     // f u
OpGenerator copySignGenerator(); // f b
OpGenerator cosGenerator();      // f u
OpGenerator sinGenerator();
OpGenerator ctlzGenerator(); // i u
OpGenerator cttzGenerator();
OpGenerator ctpopGenerator();
OpGenerator erfGenerator(); // f u // What's the semantic of this op?
OpGenerator expGenerator();
OpGenerator exp2Generator();
OpGenerator expm1Generator();
OpGenerator floorGenerator();
OpGenerator fmaGenerator();   // f t
// OpGenerator ipowiGenerator(); // i b
OpGenerator logGenerator();   // f u
OpGenerator log10Generator();
OpGenerator log1pGenerator();
OpGenerator log2Generator();
OpGenerator powfGenerator();  // f b
OpGenerator rsqrtGenerator(); // f u
OpGenerator sqrtGenerator();
OpGenerator tanGenerator();
OpGenerator tanhGenerator();
OpGenerator roundEvenGenerator();
OpGenerator roundGenerator();
OpGenerator truncGenerator();
OpGenerator fpowiGenerator(); // f,i

/*------------ func generators -----------*/
OpGenerator funcGenerator();
OpGenerator callGenerator();
OpGenerator returnGenerator();

/*-------------- scf generators -------------------*/
OpGenerator scfIfGenerator();
OpGenerator executeRegionGenerator();
OpGenerator scfForGenerator();
OpGenerator indexSwitchGenerator();
OpGenerator scfWhileGenerator();
OpGenerator scfForEachThreadGenerator(); // deprecated
OpGenerator scfParallelGenerator();

/*-------------- affine generators ------------*/
OpGenerator affineApplyGenerator();
OpGenerator affineForGenerator();
OpGenerator affineIfGenerator();
OpGenerator affineLoadGenerator();
OpGenerator affineStoreGenerator();
OpGenerator affineMinGenerator();
OpGenerator affineMaxGenerator();
OpGenerator affineParallelGenerator(); // error: 'affine.parallel' op a
// reduction must be specified for each output
OpGenerator affinePrefetchGenerator();
OpGenerator affineVectorLoadGenerator();
OpGenerator affineVectorStoreGenerator();
// OpGenerator affineDMAStartGenerator();
// OpGenerator affineDMAWaitGenerator();

/*-------------- vector generators -------------*/
OpGenerator vectorBroadcastGenerator();
OpGenerator vectorBitCastGenerator();
OpGenerator vectorConstantMaskGenerator();
OpGenerator vectorCreateMaskGenerator();
OpGenerator vectorExpandLoadGenerator();
OpGenerator vectorExtractGenerator();
OpGenerator vectorExtractElementGenerator();
OpGenerator vectorExtractStridedSliceGenerator();
OpGenerator vectorFMAGenerator();
OpGenerator vectorInsertElementGenerator();
OpGenerator vectorInsertGenerator();
OpGenerator vectorInsertStridedSliceGenerator();
OpGenerator vectorLoadGenerator();
OpGenerator vectorMaskGenerator();
OpGenerator vectorMaskedLoadGenerator();
OpGenerator vectorStoreGenerator();
OpGenerator vectorMaskedStoreGenerator();
OpGenerator vectorPrintGenerator();
OpGenerator vectorReductionGenerator();
OpGenerator vectorReshapeGenerator(); // TODO
OpGenerator vectorScanGenerator();
OpGenerator vectorScatterGenerator();
OpGenerator vectorShapeCastGenerator();
OpGenerator vectorShuffleGenerator();
OpGenerator vectorSplatGenerator();
OpGenerator vectorTypeCastGenerator();
OpGenerator vectorTransposeGenerator();
// OpGenerator vectorCompressStoreGenerator();
// OpGenerator vectorContractGenerator();
// OpGenerator vectorFlatTransposeGenerator();
// OpGenerator vectorGatherGenerator();
// OpGenerator vectorMatrixMultiplyGenerator();
// OpGenerator vectorMultiReductionGenerator();
// OpGenerator vectorOuterProductGenerator();
// OpGenerator vectorTransferReadGenerator();
// OpGenerator vectorTransferWriteGenerator();
// OpGenerator vectorWarpExecuteOnLane0Op();

/*-------------- tensor generators -------------*/
OpGenerator tensorCastGenerator();
OpGenerator tensorCollapseShapeGenerator();
OpGenerator tensorDimGenerator();
OpGenerator tensorEmptyGenerator();
OpGenerator tensorExpandShapeGenerator();
OpGenerator tensorExtractGenerator();
OpGenerator tensorExtractSliceGenerator();
OpGenerator tensorFromElementsGenerator();
OpGenerator tensorGatherGenerator();
OpGenerator tensorInsertGenerator();
OpGenerator tensorInsertSliceGenerator();
OpGenerator tensorPadGenerator();
OpGenerator tensorParallelInsertSliceGenerator();
OpGenerator tensorRankGenerator();
OpGenerator tensorReshapeGenerator();
OpGenerator tensorSplatGenerator();
// OpGenerator tensorGenerateGenerator();
// OpGenerator tensorPackGenerator();
// OpGenerator tensorScatterGenerator();
// OpGenerator tensorUnpackGenerator();

/*-------------- index generators -------------*/
OpGenerator indexAddGenerator();
OpGenerator indexAndGenerator();
OpGenerator indexBoolConstantGenerator();
OpGenerator indexCastSGenerator();
OpGenerator indexCastUGenerator();
OpGenerator indexCeilDivSGenerator();
OpGenerator indexCeilDivUGenerator();
OpGenerator indexCmpGenerator();
OpGenerator indexConstantGenerator();
OpGenerator indexDivSGenerator();
OpGenerator indexDivUGenerator();
OpGenerator indexFloorDivSGenerator();
OpGenerator indexMaxSGenerator();
OpGenerator indexMaxUGenerator();
OpGenerator indexMulGenerator();
OpGenerator indexOrGenerator();
OpGenerator indexRemSGenerator();
OpGenerator indexRemUGenerator();
OpGenerator indexShLGenerator();
OpGenerator indexShrSGenerator();
OpGenerator indexShrUGenerator();
OpGenerator indexSizeOfGenerator();
OpGenerator indexSubGenerator();
OpGenerator indexXorGenerator();

/*-------------- SPIRV generators -------------*/
/*OpGenerator spirvBitCountGenerator();
OpGenerator spirvBitReverseGenerator();
OpGenerator spirvFNegateGenerator();
OpGenerator spirvIsInfGenerator();
OpGenerator spirvIsNanGenerator();
OpGenerator spirvLogicalNotGenerator();
OpGenerator spirvNotGenerator();
OpGenerator spirvSNegateGenerator();
OpGenerator spirvBitFieldInsertGenerator();
OpGenerator spirvBitFieldSExtractGenerator();
OpGenerator spirvBitFieldUExtractGenerator();
OpGenerator spirvBitwiseAndGenerator();
OpGenerator spirvBitwiseOrGenerator();
OpGenerator spirvBitwiseXorGenerator();
OpGenerator spirvCLCeilGenerator();
OpGenerator spirvCLCosGenerator();
OpGenerator spirvCLErfGenerator();
OpGenerator spirvCLExpGenerator();
OpGenerator spirvCLFAbsGenerator();
OpGenerator spirvCLFloorGenerator();
OpGenerator spirvCLLogGenerator();
OpGenerator spirvCLRintGenerator();
OpGenerator spirvCLRoundGenerator();
OpGenerator spirvCLRsqrtGenerator();
OpGenerator spirvCLSinGenerator();
OpGenerator spirvCLSqrtGenerator();
OpGenerator spirvCLTanhGenerator();
OpGenerator spirvBitwiseOrGenerator();
OpGenerator spirvBitwiseXorGenerator();
OpGenerator spirvCLCeilGenerator();
OpGenerator spirvCLCosGenerator();
OpGenerator spirvCLErfGenerator();
OpGenerator spirvCLExpGenerator();
OpGenerator spirvCLFAbsGenerator();
OpGenerator spirvCLFloorGenerator();
OpGenerator spirvCLLogGenerator();
OpGenerator spirvCLRintGenerator();
OpGenerator spirvCLRoundGenerator();
OpGenerator spirvCLRsqrtGenerator();
OpGenerator spirvCLSinGenerator();
OpGenerator spirvCLSqrtGenerator();
OpGenerator spirvCLTanhGenerator();
OpGenerator spirvAccessChainGenerator();
OpGenerator spirvAddressOfGenerator();
OpGenerator spirvAtomicAndGenerator();
OpGenerator spirvAtomicCompareExchangeGenerator();
OpGenerator spirvAtomicCompareExchangeWeakGenerator();
OpGenerator spirvAtomicExchangeGenerator();
OpGenerator spirvAtomicIAddGenerator();
OpGenerator spirvAtomicIDecrementGenerator();
OpGenerator spirvAtomicIIncrementGenerator();
OpGenerator spirvAtomicISubGenerator();
OpGenerator spirvAtomicOrGenerator();
OpGenerator spirvAtomicSMaxGenerator();
OpGenerator spirvAtomicSMinGenerator();
OpGenerator spirvAtomicUMaxGenerator();
OpGenerator spirvAtomicUMinGenerator();
OpGenerator spirvAtomicXorGenerator();
OpGenerator spirvBitcastGenerator();
OpGenerator spirvBranchConditionalGenerator();
OpGenerator spirvBranchGenerator();
OpGenerator spirvCLFMaxGenerator();
OpGenerator spirvCLFMinGenerator();
OpGenerator spirvCLFmaGenerator();
OpGenerator spirvCLPowGenerator();
OpGenerator spirvCLSAbsGenerator();
OpGenerator spirvCLSMaxGenerator();
OpGenerator spirvCLSMinGenerator();
OpGenerator spirvCLUMaxGenerator();
OpGenerator spirvCLUMinGenerator();
OpGenerator spirvCLPrintfGenerator();
OpGenerator spirvCompositeConstructGenerator();
OpGenerator spirvCompositeExtractGenerator();
OpGenerator spirvCompositeInsertGenerator();
OpGenerator spirvConstantGenerator();
OpGenerator spirvControlBarrierGenerator();
OpGenerator spirvConvertFToSGenerator();
OpGenerator spirvConvertFToUGenerator();
OpGenerator spirvConvertPtrToUGenerator();
OpGenerator spirvConvertSToFGenerator();
OpGenerator spirvConvertUToFGenerator();
OpGenerator spirvConvertUToPtrGenerator();
OpGenerator spirvCopyMemoryGenerator();
OpGenerator spirvEXTAtomicFAddGenerator();
OpGenerator spirvEntryPointGenerator();
OpGenerator spirvExecutionModeGenerator();
OpGenerator spirvFAddGenerator();
OpGenerator spirvFConvertGenerator();
OpGenerator spirvFDivGenerator();
OpGenerator spirvFModGenerator();
OpGenerator spirvFMulGenerator();
OpGenerator spirvFOrdEqualGenerator();
OpGenerator spirvFOrdGreaterThanEqualGenerator();
OpGenerator spirvFOrdGreaterThanGenerator();
OpGenerator spirvFOrdLessThanEqualGenerator();
OpGenerator spirvFOrdLessThanGenerator();
OpGenerator spirvFOrdNotEqualGenerator();
OpGenerator spirvFUnordEqualGenerator();
OpGenerator spirvFUnordGreaterThanEqualGenerator();
OpGenerator spirvFUnordGreaterThanGenerator();
OpGenerator spirvFUnordLessThanEqualGenerator();
OpGenerator spirvFUnordLessThanGenerator();
OpGenerator spirvFUnordNotEqualGenerator();
OpGenerator spirvIEqualGenerator();
OpGenerator spirvINotEqualGenerator();
OpGenerator spirvLogicalAndGenerator();
OpGenerator spirvLogicalOrGenerator();
OpGenerator spirvLogicalEqualGenerator();
OpGenerator spirvLogicalNotEqualGenerator();
OpGenerator spirvSGreaterThanEqualGenerator();
OpGenerator spirvSGreaterThanGenerator();
OpGenerator spirvSLessThanEqualGenerator();
OpGenerator spirvSLessThanGenerator();
OpGenerator spirvUGreaterThanEqualGenerator();
OpGenerator spirvUGreaterThanGenerator();
OpGenerator spirvULessThanEqualGenerator();
OpGenerator spirvULessThanGenerator();
OpGenerator spirvUnorderedGenerator();
OpGenerator spirvGLAcosGenerator();
OpGenerator spirvGLAsinGenerator();
OpGenerator spirvGLAtanGenerator();
OpGenerator spirvGLCeilGenerator();
OpGenerator spirvGLCosGenerator();
OpGenerator spirvGLCoshGenerator();
OpGenerator spirvGLExpGenerator();
OpGenerator spirvGLFAbsGenerator();
OpGenerator spirvGLFSignGenerator();
OpGenerator spirvGLFloorGenerator();
OpGenerator spirvGLInverseSqrtGenerator();
OpGenerator spirvGLLogGenerator();
OpGenerator spirvGLRoundEvenGenerator();
OpGenerator spirvGLRoundGenerator();
OpGenerator spirvGLSinGenerator();
OpGenerator spirvGLSinhGenerator();
OpGenerator spirvGLSqrtGenerator();
OpGenerator spirvGLTanGenerator();
OpGenerator spirvGLTanhGenerator();
OpGenerator spirvGLFClampGenerator();
OpGenerator spirvGLFMaxGenerator();
OpGenerator spirvGLFMinGenerator();
OpGenerator spirvGLFMixGenerator();
OpGenerator spirvGLFindUMsbGenerator();
OpGenerator spirvGLFmaGenerator();
OpGenerator spirvGLLdexpGenerator();
OpGenerator spirvGLPowGenerator();
OpGenerator spirvGLSAbsGenerator();
OpGenerator spirvGLSClampGenerator();
OpGenerator spirvGLSMaxGenerator();
OpGenerator spirvGLSMinGenerator();
OpGenerator spirvGLSSignGenerator();
OpGenerator spirvGLUClampGenerator();
OpGenerator spirvGLUMaxGenerator();
OpGenerator spirvGLUMinGenerator();
OpGenerator spirvFRemGenerator();
OpGenerator spirvFSubGenerator();
OpGenerator spirvFuncGenerator();
OpGenerator spirvFunctionCallGenerator();
OpGenerator spirvGLFrexpStructGenerator();
OpGenerator spirvGenericCastToPtrExplicitGenerator();
OpGenerator spirvGenericCastToPtrGenerator();
OpGenerator spirvGlobalVariableGenerator();
OpGenerator spirvGroupBroadcastGenerator();
OpGenerator spirvGroupFAddGenerator();
OpGenerator spirvGroupFMaxGenerator();
OpGenerator spirvGroupFMinGenerator();
OpGenerator spirvGroupFMulKHRGenerator();
OpGenerator spirvGroupIAddGenerator();
OpGenerator spirvGroupIMulKHRGenerator();
OpGenerator spirvGroupNonUniformBallotGenerator();
OpGenerator spirvGroupNonUniformBroadcastGenerator();
OpGenerator spirvGroupNonUniformElectGenerator();
OpGenerator spirvGroupNonUniformFAddGenerator();
OpGenerator spirvGroupNonUniformFMaxGenerator();
OpGenerator spirvGroupNonUniformFMinGenerator();
OpGenerator spirvGroupNonUniformFMulGenerator();
OpGenerator spirvGroupNonUniformIAddGenerator();
OpGenerator spirvGroupNonUniformIMulGenerator();
OpGenerator spirvGroupNonUniformSMaxGenerator();
OpGenerator spirvGroupNonUniformSMinGenerator();
OpGenerator spirvGroupNonUniformShuffleDownGenerator();
OpGenerator spirvGroupNonUniformShuffleGenerator();
OpGenerator spirvGroupNonUniformShuffleUpGenerator();
OpGenerator spirvGroupNonUniformShuffleXorGenerator();
OpGenerator spirvGroupNonUniformUMaxGenerator();
OpGenerator spirvGroupNonUniformUMinGenerator();
OpGenerator spirvGroupSMaxGenerator();
OpGenerator spirvGroupSMinGenerator();
OpGenerator spirvGroupUMaxGenerator();
OpGenerator spirvGroupUMinGenerator();
OpGenerator spirvIAddCarryGenerator();
OpGenerator spirvIAddGenerator();
OpGenerator spirvIMulGenerator();
OpGenerator spirvINTELConvertBF16ToFGenerator();
OpGenerator spirvINTELConvertFToBF16Generator();
OpGenerator spirvINTELJointMatrixLoadGenerator();
OpGenerator spirvINTELJointMatrixMadGenerator();
OpGenerator spirvINTELJointMatrixStoreGenerator();
OpGenerator spirvINTELJointMatrixWorkItemLengthGenerator();
OpGenerator spirvINTELSubgroupBlockReadGenerator();
OpGenerator spirvINTELSubgroupBlockWriteGenerator();
OpGenerator spirvISubBorrowGenerator();
OpGenerator spirvISubGenerator();
OpGenerator spirvImageDrefGatherGenerator();
OpGenerator spirvImageGenerator();
OpGenerator spirvImageQuerySizeGenerator();
OpGenerator spirvInBoundsPtrAccessChainGenerator();
OpGenerator spirvKHRAssumeTrueGenerator();
OpGenerator spirvKHRCooperativeMatrixLengthGenerator();
OpGenerator spirvKHRCooperativeMatrixLoadGenerator();
OpGenerator spirvKHRCooperativeMatrixStoreGenerator();
OpGenerator spirvKHRSubgroupBallotGenerator();
OpGenerator spirvLoadGenerator();
OpGenerator spirvLoopGenerator();
OpGenerator spirvMatrixTimesMatrixGenerator();
OpGenerator spirvMatrixTimesScalarGenerator();
OpGenerator spirvMemoryBarrierGenerator();
OpGenerator spirvMergeGenerator();
OpGenerator spirvModuleGenerator();
OpGenerator spirvNVCooperativeMatrixLengthGenerator();
OpGenerator spirvNVCooperativeMatrixLoadGenerator();
OpGenerator spirvNVCooperativeMatrixMulAddGenerator();
OpGenerator spirvNVCooperativeMatrixStoreGenerator();
OpGenerator spirvOrderedGenerator();
OpGenerator spirvPtrAccessChainGenerator();
OpGenerator spirvPtrCastToGenericGenerator();
OpGenerator spirvReferenceOfGenerator();
OpGenerator spirvReturnGenerator();
OpGenerator spirvReturnValueGenerator();
OpGenerator spirvSConvertGenerator();
OpGenerator spirvSDivGenerator();
OpGenerator spirvSDotAccSatGenerator();
OpGenerator spirvSDotGenerator();
OpGenerator spirvSModGenerator();
OpGenerator spirvSMulExtendedGenerator();
OpGenerator spirvSRemGenerator();
OpGenerator spirvSUDotAccSatGenerator();
OpGenerator spirvSUDotGenerator();
OpGenerator spirvSelectGenerator();
OpGenerator spirvSelectionGenerator();
OpGenerator spirvShiftLeftLogicalGenerator();
OpGenerator spirvShiftRightArithmeticGenerator();
OpGenerator spirvShiftRightLogicalGenerator();
OpGenerator spirvSpecConstantCompositeGenerator();
OpGenerator spirvSpecConstantGenerator();
OpGenerator spirvSpecConstantOperationGenerator();
OpGenerator spirvStoreGenerator();
OpGenerator spirvTransposeGenerator();
OpGenerator spirvUConvertGenerator();
OpGenerator spirvUDivGenerator();
OpGenerator spirvUDotAccSatGenerator();
OpGenerator spirvUDotGenerator();
OpGenerator spirvUModGenerator();
OpGenerator spirvUMulExtendedGenerator();
OpGenerator spirvUndefGenerator();
OpGenerator spirvUnreachableGenerator();
OpGenerator spirvVariableGenerator();
OpGenerator spirvVectorExtractDynamicGenerator();
OpGenerator spirvVectorInsertDynamicGenerator();
OpGenerator spirvVectorShuffleGenerator();
OpGenerator spirvVectorTimesScalarGenerator();
OpGenerator spirvYieldGenerator();
*/
/*-------------- tosa generators -------------*/
// OpGenerator tosaAbsGenerator();
// OpGenerator tosaAddGenerator();
// // OpGenerator tosaApplyScaleGenerator();
// OpGenerator tosaArgMaxGenerator();
// // OpGenerator tosaArithmeticRightShiftGenerator();
// // OpGenerator tosaBitwiseAndGenerator();
// // OpGenerator tosaBitwiseNotGenerator();
// // OpGenerator tosaBitwiseOrGenerator();
// // OpGenerator tosaBitwiseXorGenerator();
// // OpGenerator tosaCastGenerator();
// // OpGenerator tosaCeilGenerator();
// OpGenerator tosaIdentityOpGenerator();
// // OpGenerator tosaLogOpGenerator();
// // OpGenerator tosaLogicalAndOpGenerator();
// // OpGenerator tosaLogicalLeftShiftOpGenerator();
// // OpGenerator tosaLogicalNotOpGenerator();
// // OpGenerator tosaLogicalOrOpGenerator();
// // OpGenerator tosaLogicalRightShiftOpGenerator();
// // OpGenerator tosaLogicalXorOpGenerator();
// // OpGenerator tosaMaxPool2dOpGenerator();
// OpGenerator tosaMaximumOpGenerator();
// OpGenerator tosaMinimumOpGenerator();
// OpGenerator tosaMulOpGenerator();
// // OpGenerator tosaPowOpGenerator();
// // OpGenerator tosaRFFT2dOpGenerator();
// // OpGenerator tosaReciprocalOpGenerator();
// //OpGenerator tosaReduceAllOpGenerator();
// //OpGenerator tosaReduceAnyOpGenerator();
// //OpGenerator tosaReduceMaxOpGenerator();
// //OpGenerator tosaReduceMinOpGenerator();
// //OpGenerator tosaReduceProdOpGenerator();
// //OpGenerator tosaReduceSumOpGenerator();
// // OpGenerator tosaRescaleOpGenerator();
// // OpGenerator tosaReshapeOpGenerator();
// // OpGenerator tosaResizeOpGenerator();
// //OpGenerator tosaReverseOpGenerator();
// // OpGenerator tosaRsqrtOpGenerator();
// // OpGenerator tosaScatterOpGenerator();
// // OpGenerator tosaSelectOpGenerator();
// // OpGenerator tosaSigmoidOpGenerator();
// OpGenerator tosaSliceOpGenerator();
// OpGenerator tosaSubOpGenerator();
// // OpGenerator tosaTableOpGenerator();
// // OpGenerator tosaTanhOpGenerator();
// OpGenerator tosaTileOpGenerator();
// // OpGenerator tosaTransposeOpGenerator();

// generator registration
static std::map<std::string, OpGen> operators = {
    {"func.func", OpGen("func.func", funcGenerator())},
    {"func.call", OpGen("func.call", callGenerator())},
    {"linalg.matmul", OpGen("linalg.matmul", linalgMatMulGenerator())},
    {"linalg.generic", OpGen("linalg.generic", linalgGenericGenerator())},
    {"linalg.map", OpGen("linalg.map", linalgMapGenerator())},
    {"linalg.copy", OpGen("linalg.copy", linalgCopyGenerator())},
    {"linalg.transpose", OpGen("linalg.transpose", linalgTransposeGenerator())},
    {"linalg.broadcast", OpGen("linalg.broadcast", linalgBroadCastGenerator())},
    {"linalg.reduce", OpGen("linalg.reduce", linalgReduceGenerator())},
    // {"linalg.dot", OpGen("linalg.dot", linalgDotGenerator())},
    {"memref.alloca", OpGen("memref.alloca", allocaGenerator())},
    // {"memref.alloca_scope", OpGen("memref.alloca_scope", allocaScopeGenerator())},
    {"memref.assume_alignment", OpGen("memref.assume_alignment", assumeAlignmentGenerator())},
    {"memref.alloc", OpGen("memref.alloc", allocGenerator())},
    // {"memref.atomic_rmw", OpGen("memref.atomic_rmw", atomicRMWGenerator())},
    {"memref.cast", OpGen("memref.cast", memrefCastGenerator())},
    // {"memref.realloc", OpGen("memref.realloc", reallocGenerator())},
    // {"memref.tensor_store", OpGen("memref.tensor_store", tensorStoreGenerator())},
    {"memref.store", OpGen("memref.store", memrefStoreGenerator())},
    {"memref.load", OpGen("memref.load", memrefLoadGenerator())},
    {"memref.copy", OpGen("memref.copy", memrefCopyGenerator())},
    {"arith.addf", OpGen("arith.addf", addFGenerator())},
    {"arith.addi", OpGen("arith.addi", addIGenerator())},
    {"arith.andi", OpGen("arith.andi", andIGenerator())},
    {"arith.cmpf", OpGen("arith.cmpf", cmpFGenerator())},
    {"arith.cmpi", OpGen("arith.cmpi", cmpIGenerator())},
    {"arith.ceildivsi", OpGen("arith.ceildivsi", ceilDivSIGenerator())},
    {"arith.constant", OpGen("arith.constant", constantGenerator())},
    {"arith.divf", OpGen("arith.divf", divFGenerator())},
    {"arith.divsi", OpGen("arith.divsi", divSIGenerator())},
    {"arith.divui", OpGen("arith.divui", divUIGenerator())},
    {"arith.floordivsi", OpGen("arith.floordivsi", floorDivSIGenerator())},
    {"arith.maxf", OpGen("arith.maxf", maxFGenerator())},
    {"arith.maxsi", OpGen("arith.maxsi", maxSIGenerator())},
    {"arith.maxui", OpGen("arith.maxui", maxUIGenerator())},
    {"arith.minf", OpGen("arith.minf", minFGenerator())},
    {"arith.minsi", OpGen("arith.minsi", minSIGenerator())},
    {"arith.minui", OpGen("arith.minui", minUIGenerator())},
    {"arith.mulf", OpGen("arith.mulf", mulFGenerator())},
    {"arith.muli", OpGen("arith.muli", mulIGenerator())},
    {"arith.negf", OpGen("arith.negf", negFGenerator())},
    {"arith.ori", OpGen("arith.ori", orIGenerator())},
    {"arith.remf", OpGen("arith.remf", remFGenerator())},
    {"arith.remsi", OpGen("arith.remsi", remSIGenerator())},
    {"arith.remui", OpGen("arith.remui", remUIGenerator())},
    {"arith.shli", OpGen("arith.shli", shlIGenerator())},
    {"arith.shrsi", OpGen("arith.shrsi", shrSIGenerator())},
    {"arith.shrui", OpGen("arith.shrui", shrUIGenerator())},
    {"arith.subf", OpGen("arith.subf", subFGenerator())},
    {"arith.subi", OpGen("arith.subi", subIGenerator())},
    {"arith.xori", OpGen("arith.xori", xorIGenerator())},
    {"math.absf", OpGen("math.absf", absFGenerator())},
    {"math.absi", OpGen("math.absi", absIGenerator())},
    {"math.atan", OpGen("math.atan", atanGenerator())},
    {"math.atan2", OpGen("math.atan2", atan2Generator())},
    {"math.ceil", OpGen("math.ceil", ceilGenerator())},
    {"math.copySign", OpGen("math.copySign", copySignGenerator())},
    {"math.cos", OpGen("math.cos", cosGenerator())},
    {"math.ctlz", OpGen("math.ctlz", ctlzGenerator())},
    {"math.cttz", OpGen("math.cttz", cttzGenerator())},
    {"math.ctpop", OpGen("math.ctpop", ctpopGenerator())},
    {"math.exp", OpGen("math.exp", expGenerator())},
    {"math.exp2", OpGen("math.exp2", exp2Generator())},
    {"math.expm1", OpGen("math.expm1", expm1Generator())},
    {"math.floor", OpGen("math.floor", floorGenerator())},
    {"math.fma", OpGen("math.fma", fmaGenerator())},
    // {"math.ipowi", OpGen("math.ipowi", ipowiGenerator())},
    {"math.log", OpGen("math.log", logGenerator())},
    {"math.log10", OpGen("math.log10", log10Generator())},
    {"math.log1p", OpGen("math.log1p", log1pGenerator())},
    {"math.log2", OpGen("math.log2", log2Generator())},
    {"math.powf", OpGen("math.powf", powfGenerator())},
    {"math.rsqrt", OpGen("math.rsqrt", rsqrtGenerator())},
    {"math.sqrt", OpGen("math.sqrt", sqrtGenerator())},
    {"math.tan", OpGen("math.tan", tanGenerator())},
    {"math.tanh", OpGen("math.tanh", tanhGenerator())},
    {"math.roundeven", OpGen("math.roundeven", roundEvenGenerator())},
    {"math.round", OpGen("math.round", roundGenerator())},
    {"math.trunc", OpGen("math.trunc", truncGenerator())},
    {"math.fpowi", OpGen("math.fpowi", fpowiGenerator())},
    {"scf.if", OpGen("scf.if", scfIfGenerator())},
    {"scf.execute_region",
     OpGen("scf.execute_region", executeRegionGenerator())},
    {"scf.index_switch", OpGen("scf.index_switch", indexSwitchGenerator())},
    {"scf.while", OpGen("scf.while", scfWhileGenerator())},
    // {"scf.parallel", OpGen("scf.parallel", scfParallelGenerator())},
    {"affine.apply", OpGen("affine.apply", affineApplyGenerator())},
    {"affine.for", OpGen("affine.for", affineForGenerator())},
    {"affine.if", OpGen("affine.if", affineIfGenerator())},
    {"affine.load", OpGen("affine.load", affineLoadGenerator())},
    {"affine.store", OpGen("affine.store", affineStoreGenerator())},
    {"affine.min", OpGen("affine.min", affineMinGenerator())},
    {"affine.max", OpGen("affine.max", affineMaxGenerator())},
    {"affine.vector_store",
     OpGen("affine.vector_store", affineVectorStoreGenerator())},
    {"affine.vector_load",
     OpGen("affine.vector_load", affineVectorLoadGenerator())},
    // {"affine.parallel", OpGen("affine.parallel", affineParallelGenerator())},
    {"vector.bitcast", OpGen("vector.bitcast", vectorBitCastGenerator())},
    {"vector.broadcast", OpGen("vector.broadcast", vectorBroadcastGenerator())},
    // {"vector.contract", OpGen("vector.contract", vectorContractGenerator())},
    // {"vector.compress_store", OpGen("vector.compress_store", vectorCompressStoreGenerator())},
    {"vector.create_mask", OpGen("vector.create_mask", vectorCreateMaskGenerator())},
    {"vector.extract", OpGen("vector.extract", vectorExtractGenerator())},
    {"vector.extract_strided_slice", OpGen("vector.extract_strided_slice", vectorExtractStridedSliceGenerator())},
    {"vector.fma", OpGen("vector.fma", vectorFMAGenerator())},
    //  {"vector.flat_transpose", OpGen("vector.flat_transpose", vectorFlatTransposeGenerator())},
    {"vector.insert_element", OpGen("vector.insert_element", vectorInsertElementGenerator())},
    {"vector.insert", OpGen("vector.insert", vectorInsertGenerator())},
    // {"vector.insert_strided_slice", OpGen("vector.insert_strided_slice", vectorInsertStridedSliceGenerator())},
    {"vector.load", OpGen("vector.load", vectorLoadGenerator())},
    // {"vector.masked_load", OpGen("vector.masked_load", vectorMaskedLoadGenerator())},
    // {"vector.matrix_multiply", OpGen("vector.matrix_multiply", vectorMatrixMultiplyGenerator())},
    // {"vector.multi_reduction", OpGen("vector.multi_reduction", vectorMultiReductionGenerator())},
    // {"vector.outer_product", OpGen("vector.outer_product", vectorOuterProductGenerator())},
    // {"vector.print", OpGen("vector.print", vectorPrintGenerator())},
    // {"vector.reduction", OpGen("vector.reduction", vectorReductionGenerator())},
    {"vector.scan", OpGen("vector.scan", vectorScanGenerator())},
    // {"vector.scatter", OpGen("vector.scatter", vectorScatterGenerator())},
    {"vector.shuffle", OpGen("vector.shuffle", vectorShuffleGenerator())},
    {"vector.splat", OpGen("vector.splat", vectorSplatGenerator())},
    {"vector.transpose", OpGen("vector.transpose", vectorTransposeGenerator())},
    // {"vector.gather", OpGen("vector.gather", vectorGatherGenerator())},
    {"vector.mask", OpGen("vector.mask", vectorMaskGenerator())},
    // {"vector.transfer_read", OpGen("vector.transfer_read", vectorTransferReadGenerator())},
    // {"vector.transfer_write", OpGen("vector.transfer_write", vectorTransferWriteGenerator())},
    // {"vector.warp_execute_on_lane0", OpGen("vector.warp_execute_on_lane0", vectorWarpExecuteOnLane0Op())},


    {"tensor.cast", OpGen("tensor.cast", tensorCastGenerator())},
    {"tensor.collapse_shape",
     OpGen("tensor.collapse_shape", tensorCollapseShapeGenerator())},
    {"tensor.dim", OpGen("tensor.dim", tensorDimGenerator())},
    {"tensor.empty", OpGen("tensor.empty", tensorEmptyGenerator())},
    // {"tensor.expand_shape",
    //  OpGen("tensor.expand_shape", tensorExpandShapeGenerator())},
    {"tensor.extract", OpGen("tensor.extract", tensorExtractGenerator())},
    {"tensor.insert_slice",
     OpGen("tensor.insert_slice", tensorInsertSliceGenerator())},
    {"tensor.extract_slice",
     OpGen("tensor.extract_slice", tensorExtractSliceGenerator())},
    {"tensor.from_elements",
     OpGen("tensor.from_elements", tensorFromElementsGenerator())},
    // {"tensor.generate", OpGen("tensor.generate", tensorGenerateGenerator())},
    {"tensor.insert", OpGen("tensor.insert", tensorInsertGenerator())},
    // {"tensor.pack", OpGen("tensor.pack", tensorPackGenerator())},
    {"tensor.rank", OpGen("tensor.rank", tensorRankGenerator())},
    // {"tensor.scatter", OpGen("tensor.scatter", tensorScatterGenerator())},
    {"tensor.splat", OpGen("tensor.splat", tensorSplatGenerator())},
    // {"tensor.unpack", OpGen("tensor.unpack", tensorUnpackGenerator())},
    {"index.add", OpGen("index.add", indexAddGenerator())},
    {"index.and", OpGen("index.and", indexAndGenerator())},
    {"index.bool.constant",
     OpGen("index.bool.constant", indexBoolConstantGenerator())},
    {"index.casts", OpGen("index.casts", indexCastSGenerator())},
    {"index.castu", OpGen("index.castu", indexCastUGenerator())},
    {"index.ceildivs", OpGen("index.ceildivs", indexCeilDivSGenerator())},
    {"index.ceildivu", OpGen("index.ceildivu", indexCeilDivUGenerator())},
    {"index.constant", OpGen("index.constant", indexConstantGenerator())},
    {"index.divs", OpGen("index.divs", indexDivSGenerator())},
    {"index.divu", OpGen("index.divu", indexDivUGenerator())},
    {"index.floordivs", OpGen("index.floordivs", indexFloorDivSGenerator())},
    {"index.maxs", OpGen("index.maxs", indexMaxSGenerator())},
    {"index.maxu", OpGen("index.maxu", indexMaxUGenerator())},
    {"index.mul", OpGen("index.mul", indexMulGenerator())},
    {"index.or", OpGen("index.or", indexOrGenerator())},
    {"index.rems", OpGen("index.rems", indexRemSGenerator())},
    {"index.remu", OpGen("index.remu", indexRemUGenerator())},
    {"index.shl", OpGen("index.shl", indexShLGenerator())},
    {"index.shrs", OpGen("index.shrs", indexShrSGenerator())},
    {"index.shru", OpGen("index.shru", indexShrUGenerator())},
    {"index.sizeof", OpGen("index.sizeof", indexSizeOfGenerator())},
    {"index.sub", OpGen("index.sub", indexSubGenerator())},
    {"index.xor", OpGen("index.xor", indexXorGenerator())},
/*    {"bufferization.alloc_tensor",
     OpGen("bufferization.alloc_tensor", bufferizationCloneGenerator())},
    {"bufferization.clone",
     OpGen("bufferization.clone", bufferizationCloneGenerator())},
    {"bufferization.dealloc_tensor",
     OpGen("bufferization.dealloc_tensor", bufferizationDeallocGenerator())},
    {"bufferization.to_memref",
     OpGen("bufferization.to_memref", bufferizationToMemrefGenerator())},
    {"bufferization.to_tensor",
     OpGen("bufferization.to_tensor", bufferizationToTensorGenerator())},
     
     
    {"spirv.BitCount", OpGen("spirv.BitCount", spirvBitCountGenerator())},
    {"spirv.BitFieldInsert", OpGen("spirv.BitFieldInsert", spirvBitFieldInsertGenerator())},
    // {"spirv.BitField", OpGen("spirv.BitField", spirvBitFieldExtractGenerator(std::string)},
    {"spirv.BitReverse", OpGen("spirv.BitReverse", spirvBitReverseGenerator())},
    {"spirv.Not", OpGen("spirv.Not", spirvNotGenerator())},
    {"spirv.BitwiseAnd", OpGen("spirv.BitwiseAnd", spirvBitwiseAndGenerator())},
    {"spirv.BitwiseOr", OpGen("spirv.BitwiseOr", spirvBitwiseOrGenerator())},
    {"spirv.BitwiseXor", OpGen("spirv.BitwiseXor", spirvBitwiseXorGenerator())},
    {"spirv.CL.ceil", OpGen("spirv.CL.ceil", spirvCLCeilGenerator())},
    {"spirv.CL.cos", OpGen("spirv.CL.cos", spirvCLCosGenerator())},
    {"spirv.CL.erf", OpGen("spirv.CL.erf", spirvCLErfGenerator())},
    {"spirv.CL.exp", OpGen("spirv.CL.exp", spirvCLExpGenerator())},
    {"spirv.CL.fabs", OpGen("spirv.CL.fabs", spirvCLFAbsGenerator())},
    {"spirv.CL.floor", OpGen("spirv.CL.floor", spirvCLFloorGenerator())},
    {"spirv.CL.log", OpGen("spirv.CL.log", spirvCLLogGenerator())},
    {"spirv.CL.round", OpGen("spirv.CL.round", spirvCLRoundGenerator())},
    {"spirv.CL.rint", OpGen("spirv.CL.rint", spirvCLRintGenerator())},
    {"spirv.CL.rsqrt", OpGen("spirv.CL.rsqrt", spirvCLRsqrtGenerator())},
    {"spirv.CL.sin", OpGen("spirv.CL.sin", spirvCLSinGenerator())},
    {"spirv.CL.sqrt", OpGen("spirv.CL.sqrt", spirvCLSqrtGenerator())},
    {"spirv.CL.tanh", OpGen("spirv.CL.tanh", spirvCLTanhGenerator())},
    {"spirv.CL.sabs", OpGen("spirv.CL.sabs", spirvCLSAbsGenerator())},
    {"spirv.CL.fmax", OpGen("spirv.CL.fmax", spirvCLFMaxGenerator())},
    {"spirv.CL.fmin", OpGen("spirv.CL.fmin", spirvCLFMinGenerator())},
    {"spirv.CL.pow", OpGen("spirv.CL.pow", spirvCLPowGenerator())},
    {"spirv.CL.smax", OpGen("spirv.CL.smax", spirvCLSMaxGenerator())},
    {"spirv.CL.smin", OpGen("spirv.CL.smin", spirvCLSMinGenerator())},
    {"spirv.CL.umax", OpGen("spirv.CL.umax", spirvCLUMaxGenerator())},
    {"spirv.CL.umin", OpGen("spirv.CL.umin", spirvCLUMinGenerator())},
    {"spirv.FNegate", OpGen("spirv.FNegate", spirvFNegateGenerator())},
    {"spirv.FOrdEqual", OpGen("spirv.FOrdEqual", spirvFOrdEqualGenerator())},
    {"spirv.FOrdLessThanEqual", OpGen("spirv.FOrdLessThanEqual", spirvFOrdLessThanEqualGenerator())},
    {"spirv.FOrdLessThan", OpGen("spirv.FOrdLessThan", spirvFOrdLessThanGenerator())},
    {"spirv.FOrdNotEqual", OpGen("spirv.FOrdNotEqual", spirvFOrdNotEqualGenerator())},
    {"spirv.FUnordEqual", OpGen("spirv.FUnordEqual", spirvFUnordEqualGenerator())},
    {"spirv.FUnordGreaterThanEqual", OpGen("spirv.FUnordGreaterThanEqual", spirvFUnordGreaterThanEqualGenerator())},
    {"spirv.FUnordGreaterThan", OpGen("spirv.FUnordGreaterThan", spirvFUnordGreaterThanGenerator())},
    {"spirv.FUnordLessThanEqual", OpGen("spirv.FUnordLessThanEqual", spirvFUnordLessThanEqualGenerator())},
    {"spirv.FUnordLessThan", OpGen("spirv.FUnordLessThan", spirvFUnordLessThanGenerator())},
    {"spirv.FUnordNotEqual", OpGen("spirv.FUnordNotEqual", spirvFUnordNotEqualGenerator())},
    {"spirv.IEqual", OpGen("spirv.IEqual", spirvIEqualGenerator())},
    {"spirv.INotEqual", OpGen("spirv.INotEqual", spirvINotEqualGenerator())},
    {"spirv.LogicalEqual", OpGen("spirv.LogicalEqual", spirvLogicalEqualGenerator())},
    {"spirv.LogicalNot", OpGen("spirv.LogicalNot", spirvLogicalNotGenerator())},
    {"spirv.LogicalNotEqual", OpGen("spirv.LogicalNotEqual", spirvLogicalNotEqualGenerator())},
    {"spirv.LogicalAnd", OpGen("spirv.LogicalAnd", spirvLogicalAndGenerator())},
    {"spirv.LogicalOr", OpGen("spirv.LogicalOr", spirvLogicalOrGenerator())},
    {"spirv.SGreaterThanEqual", OpGen("spirv.SGreaterThanEqual", spirvSGreaterThanEqualGenerator())},
    {"spirv.SGreaterThan", OpGen("spirv.SGreaterThan", spirvSGreaterThanGenerator())},
    {"spirv.SLessThanEqual", OpGen("spirv.SLessThanEqual", spirvSLessThanEqualGenerator())},
    {"spirv.SLessEqual", OpGen("spirv.SLessEqual", spirvSLessThanGenerator())},
    {"spirv.UGreaterThanEqual", OpGen("spirv.UGreaterThanEqual", spirvUGreaterThanEqualGenerator())},
    {"spirv.UGreaterThan", OpGen("spirv.UGreaterThan", spirvUGreaterThanGenerator())},
    {"spirv.ULessThanEqual", OpGen("spirv.ULessThanEqual", spirvULessThanEqualGenerator())},
    {"spirv.ULessThan", OpGen("spirv.ULessThan", spirvULessThanGenerator())},
    {"spirv.Unordered", OpGen("spirv.Unordered", spirvUnorderedGenerator())},
    {"spirv.GL.Acos", OpGen("spirv.GL.Acos", spirvGLAcosGenerator())},
    {"spirv.GL.Asin", OpGen("spirv.GL.Asin", spirvGLAsinGenerator())},
    {"spirv.GL.Atan", OpGen("spirv.GL.Atan", spirvGLAtanGenerator())},
    {"spirv.GL.Ceil", OpGen("spirv.GL.Ceil", spirvGLCeilGenerator())},
    {"spirv.GL.Cos", OpGen("spirv.GL.Cos", spirvGLCosGenerator())},
    {"spirv.GL.Cosh", OpGen("spirv.GL.Cosh", spirvGLCoshGenerator())},
    {"spirv.GL.Exp", OpGen("spirv.GL.Exp", spirvGLExpGenerator())},
    {"spirv.GL.FAbs", OpGen("spirv.GL.FAbs", spirvGLFAbsGenerator())},
    {"spirv.GL.FSign", OpGen("spirv.GL.FSign", spirvGLFSignGenerator())},
    {"spirv.GL.Floor", OpGen("spirv.GL.Floor", spirvGLFloorGenerator())},
    {"spirv.GL.InverseSqrt", OpGen("spirv.GL.InverseSqrt", spirvGLInverseSqrtGenerator())},
    {"spirv.GL.Log", OpGen("spirv.GL.Log", spirvGLLogGenerator())},
    {"spirv.GL.RoundEven", OpGen("spirv.GL.RoundEven", spirvGLRoundEvenGenerator())},
    {"spirv.GL.Round", OpGen("spirv.GL.Round", spirvGLRoundGenerator())},
    {"spirv.GL.Sin", OpGen("spirv.GL.Sin", spirvGLSinGenerator())},
    {"spirv.GL.Sinh", OpGen("spirv.GL.Sinh", spirvGLSinhGenerator())},
    {"spirv.GL.Sqrt", OpGen("spirv.GL.Sqrt", spirvGLSqrtGenerator())},
    {"spirv.GL.Tan", OpGen("spirv.GL.Tan", spirvGLTanGenerator())},
    {"spirv.GL.Tanh", OpGen("spirv.GL.Tanh", spirvGLTanhGenerator())},
    {"spirv.GL.FClamp", OpGen("spirv.GL.FClamp", spirvGLFClampGenerator())},
    {"spirv.GL.SAbs", OpGen("spirv.GL.SAbs", spirvGLSAbsGenerator())},
    {"spirv.GL.SSign", OpGen("spirv.GL.SSign", spirvGLSSignGenerator())},
    {"spirv.GL.FMax", OpGen("spirv.GL.FMax", spirvGLFMaxGenerator())},
    {"spirv.GL.FMin", OpGen("spirv.GL.FMin", spirvGLFMinGenerator())},
    {"spirv.GL.FMix", OpGen("spirv.GL.FMix", spirvGLFMixGenerator())},
    {"spirv.GL.FindUMsb", OpGen("spirv.GL.FindUMsb", spirvGLFindUMsbGenerator())},
    {"spirv.GL.Ldexp", OpGen("spirv.GL.Ldexp", spirvGLLdexpGenerator())},
    {"spirv.GL.Pow", OpGen("spirv.GL.Pow", spirvGLPowGenerator())},
    {"spirv.GL.SClamp", OpGen("spirv.GL.SClamp", spirvGLSClampGenerator())},
    {"spirv.GL.SMax", OpGen("spirv.GL.SMax", spirvGLSMaxGenerator())},
    {"spirv.GL.SMin", OpGen("spirv.GL.SMin", spirvGLSMinGenerator())},
    {"spirv.GL.UMax", OpGen("spirv.GL.UMax", spirvGLUMaxGenerator())},
    {"spirv.GL.UMin", OpGen("spirv.GL.UMin", spirvGLUMinGenerator())},
    {"spirv.GL.UClamp", OpGen("spirv.GL.UClamp", spirvGLFmaGenerator())},
    {"spirv.IsInf", OpGen("spirv.IsInf", spirvIsInfGenerator())},
    {"spirv.IsNan", OpGen("spirv.IsNan", spirvIsNanGenerator())},
    // {"spirv.SNegate", OpGen("spirv.SNegate", getSPIRVBoolBinaryOpGenerator(std::string)},
    {"spirv.GL.Frexp", OpGen("spirv.GL.Frexp", spirvFOrdGreaterThanGenerator())},
*/
    // // tosa dialect
    // {"tosa.abs", OpGen("tosa.abs", tosaAbsGenerator())},
    // {"tosa.add", OpGen("tosa.add", tosaAddGenerator())},
    // // {"tosa.applyScale", OpGen("tosa.applyScale", tosaApplyScaleGenerator())},
    // // {"tosa.argmax", OpGen("tosa.argmax", tosaArgMaxGenerator())},
    // // {"tosa.arithmetic_right_shift", OpGen("tosa.arithmetic_right_shift", tosaArithmeticRightShiftGenerator())},
    // // {"tosa.bitwise_and", OpGen("tosa.bitwise_and", tosaBitwiseAndGenerator())},
    // // {"tosa.bitwise_not", OpGen("tosa.bitwise_not", tosaBitwiseNotGenerator())},
    // // {"tosa.bitwise_or", OpGen("tosa.bitwise_or", tosaBitwiseOrGenerator())},
    // // {"tosa.bitwise_xor", OpGen("tosa.bitwise_xor", tosaBitwiseXorGenerator())},
    // // {"tosa.cast", OpGen("tosa.cast", tosaCastGenerator())},
    // // {"tosa.ceil", OpGen("tosa.ceil", tosaCeilGenerator())},
    // {"tosa.identity", OpGen("tosa.identity", tosaIdentityOpGenerator())},
    // // {"tosa.log", OpGen("tosa.log", tosaLogOpGenerator())},
    // // {"tosa.logical_and", OpGen("tosa.logical_and", tosaLogicalAndOpGenerator())},
    // // {"tosa.logical_left_shift", OpGen("tosa.logical_left_shift", tosaLogicalLeftShiftOpGenerator())},
    // // {"tosa.logical_not", OpGen("tosa.logical_not", tosaLogicalNotOpGenerator())},
    // // {"tosa.logical_or", OpGen("tosa.logical_or", tosaLogicalOrOpGenerator())},
    // // {"tosa.logical_right_shift", OpGen("tosa.logical_right_shift", tosaLogicalRightShiftOpGenerator())},
    // // {"tosa.logical_xor", OpGen("tosa.logical_xor", tosaLogicalXorOpGenerator())},
    // // {"tosa.max_pool2d", OpGen("tosa.max_pool2d", tosaMaxPool2dOpGenerator())},
    // {"tosa.maximum", OpGen("tosa.maximum", tosaMaximumOpGenerator())},
    // {"tosa.minimum", OpGen("tosa.minimum", tosaMinimumOpGenerator())},
    // // {"tosa.mul", OpGen("tosa.mul", tosaMulOpGenerator())},
    // // {"tosa.pow", OpGen("tosa.pow", tosaPowOpGenerator())},
    // // {"tosa.rfft2d", OpGen("tosa.rfft2d", tosaRFFT2dOpGenerator())},
    // // {"tosa.reciprocal", OpGen("tosa.reciprocal", tosaReciprocalOpGenerator())},
    // //{"tosa.reduce_all", OpGen("tosa.reduce_all", tosaReduceAllOpGenerator())},  
    // //{"tosa.reduce_any", OpGen("tosa.reduce_any", tosaReduceAnyOpGenerator())},  
    // //{"tosa.reduce_max", OpGen("tosa.reduce_max", tosaReduceMaxOpGenerator())},
    // //{"tosa.reduce_min", OpGen("tosa.reduce_min", tosaReduceMinOpGenerator())},  
    // //{"tosa.reduce_prod", OpGen("tosa.reduce_prod", tosaReduceProdOpGenerator())},  
    // //{"tosa.reduce_sum", OpGen("tosa.reduce_sum", tosaReduceSumOpGenerator())},   
    // // {"tosa.rescale", OpGen("tosa.rescale", tosaRescaleOpGenerator())},  
    // // {"tosa.reshape", OpGen("tosa.reshape", tosaReshapeOpGenerator())},  
    // // {"tosa.resize", OpGen("tosa.resize", tosaResizeOpGenerator())}, 
    // //{"tosa.reverse", OpGen("tosa.reverse", tosaReverseOpGenerator())}, 
    // // {"tosa.rsqrt", OpGen("tosa.rsqrt", tosaRsqrtOpGenerator())}, 
    // // {"tosa.scatter", OpGen("tosa.scatter", tosaScatterOpGenerator())}, 
    // // {"tosa.select", OpGen("tosa.select", tosaSelectOpGenerator())}, 
    // // {"tosa.sigmoid", OpGen("tosa.sigmoid", tosaSigmoidOpGenerator())}, 
    // // {"tosa.slice", OpGen("tosa.slice", tosaSliceOpGenerator())},
    // {"tosa.sub", OpGen("tosa.sub", tosaSubOpGenerator())},
    // // {"tosa.table", OpGen("tosa.table", tosaTableOpGenerator())},
    // // {"tosa.tanh", OpGen("tosa.tanh", tosaTanhOpGenerator())},
    // // {"tosa.tile", OpGen("tosa.tile", tosaTileOpGenerator())},
    // // {"tosa.transpose", OpGen("tosa.transpose", tosaTransposeOpGenerator())}


     };

OpGen getOpGen(std::string name);

// To be deprecated
struct OpGeneration {
  std::string op;
  OpGenerator generator;

  OpGeneration(std::string op, OpGenerator opGenerator)
      : op(op), generator(opGenerator) {}

  void apply(mlir::OpBuilder &, mlir::Location, OpRegion &);
};

inline std::set<std::string> opsForFunc = {"func.call",
                                           "linalg.matmul",
                                           "linalg.generic",
                                           "linalg.map",
                                           "linalg.copy",
                                           "linalg.transpose",
                                           "linalg.broadcast",
                                           "linalg.reduce",
                                           "linalg.dot",
                                           "memref.alloca",
                                           "memref.alloca_scope",
                                           "memref.assume_alignment",
                                           "memref.alloc",
                                           "memref.atomic_rmw",
                                           "memref.cast",
                                           "memref.realloc",
                                           "memref.tensor_store",
                                           "memref.store",
                                           "memref.load",
                                           "memref.copy",
                                           "arith.addf",
                                           "arith.addi",
                                           "arith.andi",
                                           "arith.cmpf",
                                           "arith.cmpi",
                                           "arith.ceildivsi",
                                           "arith.constant",
                                           "arith.divf",
                                           "arith.divsi",
                                           "arith.divui",
                                           "arith.floordivsi",
                                        //    "arith.maxf",
                                           "arith.maxsi",
                                           "arith.maxui",
                                        //    "arith.minf",
                                           "arith.minsi",
                                           "arith.minui",
                                           "arith.mulf",
                                           "arith.muli",
                                           "arith.negf",
                                           "arith.ori",
                                           "arith.remf",
                                           "arith.remsi",
                                           "arith.remui",
                                           "arith.shli",
                                           "arith.shrsi",
                                           "arith.shrui",
                                           "arith.subf",
                                           "arith.subi",
                                           "arith.xori",
                                           "math.absf",
                                           "math.absi",
                                           "math.atan",
                                           "math.atan2",
                                           "math.ceil",
                                           "math.copySign",
                                           "math.cos",
                                           "math.ctlz",
                                           "math.cttz",
                                           "math.ctpop",
                                           "math.exp",
                                           "math.exp2",
                                           "math.expm1",
                                           "math.floor",
                                           "math.fma",
                                           "math.ipowi",
                                           "math.log",
                                           "math.log10",
                                           "math.log1p",
                                           "math.log2",
                                           "math.powf",
                                           "math.rsqrt",
                                           "math.sqrt",
                                           "math.tan",
                                           "math.tanh",
                                           "math.roundeven",
                                           "math.round",
                                           "math.trunc",
                                           "math.fpowi",
                                           "scf.if",
                                           "scf.execute_region",
                                           "scf.index_switch",
                                           "scf.while",
                                        //    "scf.parallel",
                                           "affine.apply",
                                           "affine.for",
                                           "affine.if",
                                           "affine.load",
                                           "affine.store",
                                           "affine.min",
                                           "affine.max",
                                        //    "affine.parallel",
                                           "affine.vector_store",
                                           "affine.vector_load",
                                           "vector.bitcast",
                                           "vector.broadcast",
                                           "vector.contract",
                                           "vector.compress_store",
                                           "vector.create_mask",
                                        //    "vector.extract",
                                           "vector.extract_strided_slice",
                                           "vector.fma",
                                           "vector.flat_transpose",
                                           "vector.insert_element",
                                           "vector.insert",
                                           "vector.insert_strided_slice",
                                           "vector.load",
                                           "vector.masked_load",
                                           "vector.matrix_multiply",
                                           "vector.multi_reduction",
                                           "vector.outer_product",
                                           "vector.print",
                                           "vector.reduction",
                                        //    "vector.scan",
                                           "vector.scatter",
                                           "vector.shuffle",
                                           "vector.splat",
                                           "vector.transpose",
                                           "vector.gather",
                                           "vector.mask",
                                           "vector.transfer_read",
                                           "vector.transfer_write",
                                           "vector.warp_execute_on_lane0",
                                           "tensor.cast",
                                           "tensor.collapse_shape",
                                           "tensor.dim",
                                           "tensor.empty",
                                           "tensor.expand_shape",
                                           "tensor.extract",
                                           "tensor.insert_slice",
                                           "tensor.extract_slice",
                                           "tensor.from_elements",
                                           "tensor.generate",
                                           "tensor.insert",
                                           "tensor.pack",
                                           "tensor.rank",
                                           "tensor.scatter",
                                           "tensor.splat",
                                           "tensor.unpack",
                                           "index.add",
                                           "index.and",
                                           "index.bool.constant",
                                           "index.casts",
                                           "index.castu",
                                           "index.ceildivs",
                                           "index.ceildivu",
                                           "index.constant",
                                           "index.divs",
                                           "index.divu",
                                           "index.floordivs",
                                           "index.maxs",
                                           "index.maxu",
                                           "index.mul",
                                           "index.or",
                                           "index.rems",
                                           "index.remu",
                                           "index.shl",
                                           "index.shrs",
                                           "index.shru",
                                           "index.sizeof",
                                           "index.sub",
                                           "index.xor",
                                           "bufferization.alloc_tensor",
                                           "bufferization.clone",
                                           "bufferization.dealloc_tensor",
                                           "bufferization.to_memref",
                                           "bufferization.to_tensor",
                                           
                                        //    "spirv.BitCount",
                                        //    "spirv.BitFieldInsert",
                                        //    "spirv.BitField",
                                        //    "spirv.BitReverse",
                                        //    "spirv.Not",
                                        //    "spirv.BitwiseAnd",
                                        //    "spirv.BitwiseOr",
                                        //    "spirv.BitwiseXor",
                                        //    "spirv.CL.ceil",
                                        //    "spirv.CL.cos",
                                        //    "spirv.CL.erf",
                                        //    "spirv.CL.exp",
                                        //    "spirv.CL.fabs",
                                        //    "spirv.CL.floor",
                                        //    "spirv.CL.log",
                                        //    "spirv.CL.round",
                                        //    "spirv.CL.rint",
                                        //    "spirv.CL.rsqrt",
                                        //    "spirv.CL.sin",
                                        //    "spirv.CL.sqrt",
                                        //    "spirv.CL.tanh",
                                        //    "spirv.CL.sabs",
                                        //    "spirv.CL.fmax",
                                        //    "spirv.CL.fmin",
                                        //    "spirv.CL.pow",
                                        //    "spirv.CL.smax",
                                        //    "spirv.CL.smin",
                                        //    "spirv.CL.umax",
                                        //    "spirv.CL.umin",
                                        //    "spirv.FNegate",
                                        //    "spirv.FOrdEqual",
                                        //    "spirv.FOrdLessThanEqual",
                                        //    "spirv.FOrdLessThan",
                                        //    "spirv.FOrdNotEqual",
                                        //    "spirv.FUnordEqual",
                                        //    "spirv.FUnordGreaterThanEqual",
                                        //    "spirv.FUnordGreaterThan",
                                        //    "spirv.FUnordLessThanEqual",
                                        //    "spirv.FUnordLessThan",
                                        //    "spirv.FUnordNotEqual",
                                        //    "spirv.IEqual",
                                        //    "spirv.INotEqual",
                                        //    "spirv.LogicalEqual",
                                        //    "spirv.LogicalNot",
                                        //    "spirv.LogicalNotEqual",
                                        //    "spirv.LogicalAnd",
                                        //    "spirv.LogicalOr",
                                        //    "spirv.SGreaterThanEqual",
                                        //    "spirv.SGreaterThan",
                                        //    "spirv.SLessThanEqual",
                                        //    "spirv.SLessEqual",
                                        //    "spirv.UGreaterThanEqual",
                                        //    "spirv.UGreaterThan",
                                        //    "spirv.ULessThanEqual",
                                        //    "spirv.ULessThan",
                                        //    "spirv.Unordered",
                                        //    "spirv.GL.Acos",
                                        //    "spirv.GL.Asin",
                                        //    "spirv.GL.Atan",
                                        //    "spirv.GL.Ceil",
                                        //    "spirv.GL.Cos",
                                        //    "spirv.GL.Cosh",
                                        //    "spirv.GL.Exp",
                                        //    "spirv.GL.FAbs",
                                        //    "spirv.GL.FSign",
                                        //    "spirv.GL.Floor",
                                        //    "spirv.GL.InverseSqrt",
                                        //    "spirv.GL.Log",
                                        //    "spirv.GL.RoundEven",
                                        //    "spirv.GL.Round",
                                        //    "spirv.GL.Sin",
                                        //    "spirv.GL.Sinh",
                                        //    "spirv.GL.Sqrt",
                                        //    "spirv.GL.Tan",
                                        //    "spirv.GL.Tanh",
                                        //    "spirv.GL.FClamp",
                                        //    "spirv.GL.SAbs",
                                        //    "spirv.GL.SSign",
                                        //    "spirv.GL.FMax",
                                        //    "spirv.GL.FMin",
                                        //    "spirv.GL.FMix",
                                        //    "spirv.GL.FindUMsb",
                                        //    "spirv.GL.Ldexp",
                                        //    "spirv.GL.Pow",
                                        //    "spirv.GL.SClamp",
                                        //    "spirv.GL.SMax",
                                        //    "spirv.GL.SMin",
                                        //    "spirv.GL.UMax",
                                        //    "spirv.GL.UMin",
                                        //    "spirv.GL.UClamp",
                                        //    "spirv.IsInf",
                                        //    "spirv.IsNan",
                                        //    "spirv.SNegate",
                                        //    "spirv.GL.Frexp"
                                        
                                        "tosa.abs",
                                        "tosa.add",
                                        "tosa.applyScale",
                                        // "tosa.argmax", //i64
                                        "tosa.arithmetic_right_shift",
                                        "tosa.bitwise_and",
                                        "tosa.bitwise_not",
                                        "tosa.bitwise_or",
                                        "tosa.bitwise_xor",
                                        "tosa.cast",
                                        "tosa.ceil",
                                        "tosa.identity",
                                        "tosa.log",
                                        "tosa.logical_and",
                                        "tosa.logical_left_shift",
                                        "tosa.logical_not",
                                        "tosa.logical_or",
                                        "tosa.logical_right_shift",
                                        "tosa.logical_xor",
                                        "tosa.max_pool2d",
                                        "tosa.maximum",
                                        "tosa.minimum",
                                        // "tosa.mul",
                                        "tosa.pow",
                                        "tosa.rfft2d",
                                        "tosa.reciprocal",
                                        "tosa.reduce_all", //i64
                                        "tosa.reduce_any", //i64
                                        "tosa.reduce_max", //i64
                                        "tosa.reduce_min", //i64
                                        "tosa.reduce_prod", //i64
                                        "tosa.reduce_sum", //i64
                                        "tosa.rescale",
                                        "tosa.reshape",
                                        // "tosa.resize",
                                        "tosa.reverse", //i64
                                        "tosa.rsqrt",
                                        "tosa.scatter",
                                        "tosa.select",
                                        "tosa.sigmoid",
                                        // "tosa.slice",
                                        "tosa.sub",
                                        "tosa.table",
                                        "tosa.tanh",
                                        // "tosa.tile",
                                        "tosa.transpose"

                                           };

inline std::set<std::string> intOpsForGenericAtomicRMW = {
    "arith.addi",     "arith.andi",  "arith.cmpi",  "arith.ceildivsi",
    "arith.constant", "arith.divsi", "arith.divui", "arith.floordivsi",
    "arith.maxsi",    "arith.maxui", "arith.minsi", "arith.minui",
    "arith.muli",     "arith.ori",   "arith.remsi", "arith.remui",
    "arith.shli",     "arith.shrsi", "arith.shrui", "arith.subi",
    "arith.xori",     "math.absi",   "math.ipowi",  "math.log",
    "math.log10",     "math.fpowi"};

inline std::set<std::string> floatOpsForGenericAtomicRMW = {
    "arith.addf", "arith.cmpf",     "arith.constant", "arith.divf",
    "arith.maxf", "arith.minf",     "arith.mulf",     "arith.negf",
    "arith.remf", "arith.subf",     "math.absf",      "math.atan",
    "math.atan2", "math.ceil",      "math.copySign",  "math.cos",
    "math.ctlz",  "math.cttz",      "math.ctpop",     "math.exp",
    "math.exp2",  "math.expm1",     "math.floor",     "math.fma",
    "math.log",   "math.log10",     "math.log1p",     "math.log2",
    "math.powf",  "math.rsqrt",     "math.sqrt",      "math.tan",
    "math.tanh",  "math.roundeven", "math.round",     "math.trunc",
    "math.fpowi"};

inline std::set<std::string> maskableOps = {
    "vector.multi_reduction",
    //    "vector.reduction", // this op returns non-tensor type
    "vector.transfer_read", "vector.transfer_write"};

// Ops that have can recursively define region .
inline std::set<std::string> nonTerminatorOps = {
    "func.func",        "memref.alloca_scope",
    "scf.if",           "scf.execute_region",
    "scf.index_switch", "scf.while",
    "affine.for",       "affine.if"};

inline std::set<std::string> regionedOps = {
    "func.func",       "memref.alloca_scope",
    "scf.if",          "scf.execute_region",
    "linalg.map",      "scf.for",
    "linalg.generic",  "scf.index_switch",
    "scf.while",       "scf.parallel",
    "affine.for",      "affine.if",
    // "affine.parallel",
    "vector.warp_execute_on_lane0",
    "tensor.generate"};

inline std::set<std::string> opsForAlloca = opsForFunc;
inline std::set<std::string> opsForScfIf = opsForFunc;
inline std::set<std::string> opsForScfFor = opsForFunc;
inline std::set<std::string> opsForScfWhile = opsForFunc;
inline std::set<std::string> opsForScfForeachThread = opsForFunc;
inline std::set<std::string> opsForExecuteRegion = opsForFunc;
inline std::set<std::string> opsForIndexSwitch = opsForFunc;
inline std::set<std::string> opsForLinalgMap = opsForFunc;
inline std::set<std::string> opsForLinalgGeneric = opsForFunc;
inline std::set<std::string> opsForAffineFor = opsForFunc;
inline std::set<std::string> opsForAffineIf = opsForFunc;
inline std::set<std::string> opsForAffineParallel = opsForFunc; // TODO
inline std::set<std::string> opsForTensorGenerate = opsForFunc;
inline std::set<std::string> opsForVectorWarpExecuteOnLane0 = opsForFunc;
inline std::set<std::string> opsForLinalgReduce = opsForFunc;
inline std::set<std::string> opsForScfParallel = opsForFunc;
inline std::set<std::string> opsForScfReduce = opsForFunc;

inline std::map<std::string, std::set<std::string>> availableOpNests = {
    {"func.func", opsForFunc},
    {"memref.alloca_scope", opsForAlloca},
    {"linalg.map", opsForLinalgMap},
    {"linalg.generic", opsForLinalgGeneric},
    {"scf.if", opsForScfIf},
    {"scf.execute_region", opsForExecuteRegion},
    {"scf.for", opsForScfFor},
    {"scf.while", opsForScfWhile},
    {"scf.index_switch", opsForIndexSwitch},
    {"affine.if", opsForAffineIf},
    {"affine.for", opsForAffineFor},
    // {"affine.parallel", opsForAffineParallel},
    {"tensor.generate", opsForTensorGenerate},
    {"vector.warp_execute_on_lane0", opsForVectorWarpExecuteOnLane0}};

#endif
