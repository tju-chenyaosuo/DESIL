//===- toyc.cpp - The Toy Compiler ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the entry point for the Toy compiler.
//
//===----------------------------------------------------------------------===//
#include <random>


#include "checksum/checksum.h"

#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include <iostream>

namespace cl = llvm::cl;

static cl::opt<std::string>
    configFileName("c", cl::desc("Specify json config filename"),
                   cl::value_desc("config file name"));

static cl::opt<std::string> inputFilename(cl::Positional,
                                          cl::desc("<input toy file>"),
                                          cl::init("-"),
                                          cl::value_desc("filename"));

static cl::opt<bool> enableOpt("opt", cl::desc("Enable optimizations"));
struct Mutation {
  mlir::Operation *fixUB(mlir::Operation *op);
};

int dumpMLIR() {

  mlir::MLIRContext context;

  context.getOrLoadDialect<mlir::scf::SCFDialect>();
  context.getOrLoadDialect<mlir::linalg::LinalgDialect>();
  context.getOrLoadDialect<mlir::affine::AffineDialect>();
  context.getOrLoadDialect<mlir::arith::ArithDialect>();
  context.getOrLoadDialect<mlir::index::IndexDialect>();
  context.getOrLoadDialect<mlir::memref::MemRefDialect>();
  context.getOrLoadDialect<mlir::math::MathDialect>();
  context.getOrLoadDialect<mlir::bufferization::BufferizationDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::vector::VectorDialect>();
  context.getOrLoadDialect<mlir::spirv::SPIRVDialect>();
  context.getOrLoadDialect<mlir::tosa::TosaDialect>();

  mlir::OwningOpRef<mlir::ModuleOp> module;
  llvm::SourceMgr sourceMgr;

  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
      llvm::MemoryBuffer::getFileOrSTDIN(inputFilename);
  sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
  module = mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);
  if (!module) {
    llvm::errs() << "[main] Error can't load file " << inputFilename << "\n";
    return 3;
  }

  mlir::Operation *op = module.get();
  Mutation mp = Mutation();
  mp.fixUB(op);
  module->dump();

  return 0;
}

mlir::Operation *mainfunc;
mlir::Operation* CheckSum::mainfunc ;
mlir::Operation* BuildFuncOp(mlir::Operation *op){


  mlir::IntegerType type_16 = mlir::IntegerType::get(op->getContext(), 16);


  mlir::OpBuilder builder(op->getBlock(), op->getBlock()->begin());
    builder.setInsertionPoint(op);
    mlir::RankedTensorType tensorType = mlir::RankedTensorType::get(
        /*shape=*/{mlir::ShapedType::kDynamic}, type_16);
    llvm::SmallVector<mlir::Type> argTypes;
    argTypes.push_back(tensorType);
    llvm::SmallVector<mlir::Type> retTypes;
    retTypes.push_back(type_16);
    auto funcType = mlir::FunctionType::get(
        op->getContext(), mlir::TypeRange(argTypes), mlir::TypeRange(retTypes));

    auto newFuncOp = builder.create<mlir::func::FuncOp>(
        op->getLoc(), "checksum_i16", funcType);
    newFuncOp.addEntryBlock();
    builder.setInsertionPointToEnd(&newFuncOp.getBody().front());


    auto idx0op = builder.create<mlir::index::ConstantOp>(
      op->getLoc(), builder.getIndexType(),
      mlir::IntegerAttr::get(builder.getIndexType(), 0)); 

    auto idx1op = builder.create<mlir::index::ConstantOp>(
      op->getLoc(), builder.getIndexType(),
      mlir::IntegerAttr::get(builder.getIndexType(), 1)); 

    auto zeroconstantop = builder.create<mlir::arith::ConstantOp>(
        op->getLoc(), type_16, mlir::IntegerAttr::get(type_16, 0));

    auto dimop = builder.create<mlir::tensor::DimOp>(op->getLoc(), newFuncOp.getBody().getArguments()[0], idx1op.getResult());


    auto blockBuilder = [&](mlir::OpBuilder &b, mlir::Location loc,
                            mlir::Value iv /*loop iterator*/, mlir::ValueRange args) {

      
      auto extractop = b.create<mlir::tensor::ExtractOp>(loc, newFuncOp.getBody().getArguments()[0] , iv);
      auto addop = b.create<mlir::arith::AddIOp>(loc, args[0], extractop.getResult());


      // auto zeroconstantop = b.create<mlir::arith::ConstantOp>(
      //   loc, type_16, mlir::IntegerAttr::get(type_16, 0));
      b.create<mlir::scf::YieldOp>(op->getLoc(), addop.getResult());
    };
    auto forop = builder.create<mlir::scf::ForOp>(op->getLoc(), idx0op.getResult(), dimop.getResult(), idx1op.getResult(),
                                          llvm::ArrayRef(zeroconstantop.getResult()),
                                          blockBuilder);





    builder.create<mlir::func::ReturnOp>(op->getLoc(),
                                         forop.getResult(0));
    // llvm::outs() << newFuncOp.getNumArguments() << "\n";
    return newFuncOp;
}


CheckSum checksumValuePool;

mlir::Operation *Mutation::fixUB(mlir::Operation *op) {
  // Traverse first
  for (mlir::Region &region : op->getRegions()) {
    for (mlir::Block &block : region.getBlocks()) {
      // std::vector<mlir::Operation*> oplist;
      std::string opname = op->getName().getStringRef().str();
      if(opname == "func.func")
      {
        // mlir::Identifier funcTypeAttrName = mlir::function_like::TypeAttr::getFunctionTypeAttrName();
        // mlir::NamedAttribute attr = op->getAttr(funcTypeAttrName);
        mlir::func::FuncOp funcop = mlir::cast<mlir::func::FuncOp>(*op);
        if(funcop.getName().str() == "func1")
          mainfunc = op;
          CheckSum::mainfunc = op;
        
      }
      if (opname!="builtin.module" && opname!="func.func")
      {
        // llvm::outs()<<opname<<"\n";
        break;
      }
      for (mlir::Operation &op : block.getOperations()) {
        fixUB(&op);
        // oplist.push_back(fixUB(&op));
      }
    }
  }

  std::string opname = op->getName().getStringRef().str();
  if (opname == "tensor.collapse_shape") {
    // mlir::Operation *retop = BuildFuncOp(mainfunc);
    // mlir::func::FuncOp funcop = mlir::cast<mlir::func::FuncOp>(retop);
    // mlir::OpBuilder builder(op->getBlock(), op->getBlock()->begin());
    // builder.setInsertionPointAfter(op);
    // llvm::SmallVector<mlir::Value> arguments;
    // arguments.push_back(op->getResult(0));
    // auto newFuncOp = builder.create<mlir::func::CallOp>(
    //     op->getLoc(), funcop, mlir::ValueRange(arguments));

    // mlir::Value res = op->getResult(0);
    // llvm::outs() << res.getType().cast<mlir::RankedTensorType>().getRank()
    //              << "\n";
  }

  if (opname == "func.return") {
    mlir::OpBuilder builder(op->getBlock(), op->getBlock()->begin());
    builder.setInsertionPoint(op);
    checksumValuePool.calCheckSumAll(op->getLoc(), builder);
  }

  // if (opname == "memref.alloc") {
  //   mlir::OpBuilder builder(op->getBlock(), op->getBlock()->begin());
  //   builder.setInsertionPointAfter(op);
  //   checksumValuePool.calMemrefCheckSumByType(op->getLoc(), builder, builder.getI64Type(), "i64", op->getResult(0));
  // }


  //  information->insert(getOperationName(op));
  for(int i = 0; i < op->getNumResults() ; i++)
    checksumValuePool.addValue( op->getResult(i) );
  return op;
}

int main(int argc, char **argv) {
  // Register any command line options.
  mlir::registerAsmPrinterCLOptions();
  mlir::registerMLIRContextCLOptions();
  mlir::registerPassManagerCLOptions();
  //  mlir::init
  cl::ParseCommandLineOptions(argc, argv, "toy compiler\n");
  llvm::InitLLVM(argc, argv);

  return dumpMLIR();

  return 0;
}
