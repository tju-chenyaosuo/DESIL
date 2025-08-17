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

#include "smith/DiversityCriteria.h"
#include "smith/ExpSetting.h"
#include "smith/config.h"
#include "toy/Dialect.h"
#include "toy/MLIRGen.h"
#include "toy/Parser.h"
#include "toy/Passes.h"
#include "smith/MLIRSmith.h"

#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
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

using namespace toy;
namespace cl = llvm::cl;

static cl::opt<std::string>
    configFileName("c", cl::desc("Specify json config filename"),
                   cl::value_desc("config file name"));

static cl::opt<bool> isDiverse("d", cl::desc("Generate diversely"));

static cl::opt<std::string> dFileName("dfile", cl::desc("Generate diversely"));

static cl::opt<std::string> inputFilename(cl::Positional,
                                          cl::desc("<input toy file>"),
                                          cl::init("-"),
                                          cl::value_desc("filename"));

namespace {
enum InputType { Toy, MLIR };
} // namespace
static cl::opt<enum InputType> inputType(
    "x", cl::init(Toy), cl::desc("Decided the kind of output desired"),
    cl::values(clEnumValN(Toy, "toy", "load the input file as a Toy source.")),
    cl::values(clEnumValN(MLIR, "mlir",
                          "load the input file as an MLIR file")));

namespace {
enum Action { None, DumpConfig, DumpAST, DumpMLIR, DumpMLIRAffine, MLIRSmith };
} // namespace
static cl::opt<enum Action> emitAction(
    "emit", cl::desc("Select the kind of output desired"),
    cl::values(clEnumValN(DumpAST, "ast", "output the AST dump")),
    cl::values(clEnumValN(DumpMLIR, "mlir", "output the MLIR dump")),
    cl::values(clEnumValN(DumpConfig, "config",
                          "output the configuration template")),
    cl::values(clEnumValN(DumpMLIRAffine, "mlir-affine",
                          "output the MLIR dump after affine lowering")));

static cl::opt<bool> enableOpt("opt", cl::desc("Enable optimizations"));

/// Returns a Toy AST resulting from parsing the file or a nullptr on error.
std::unique_ptr<toy::ModuleAST> parseInputFile(llvm::StringRef filename) {
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
      llvm::MemoryBuffer::getFileOrSTDIN(filename);
  if (std::error_code ec = fileOrErr.getError()) {
    llvm::errs() << "Could not open input file: " << ec.message() << "\n";
    return nullptr;
  }
  auto buffer = fileOrErr.get()->getBuffer();
  LexerBuffer lexer(buffer.begin(), buffer.end(), std::string(filename));
  Parser parser(lexer);
  return parser.parseModule();
}

int loadMLIR(llvm::SourceMgr &sourceMgr, mlir::MLIRContext &context,
             mlir::OwningOpRef<mlir::ModuleOp> &module) {
  // Handle '.toy' input to the compiler.
  if (inputType != InputType::MLIR &&
      !llvm::StringRef(inputFilename).ends_with(".mlir")) {
    auto moduleAST = parseInputFile(inputFilename);
    if (!moduleAST)
      return 6;
    module = mlirGen(context, *moduleAST);
    return !module ? 1 : 0;
  }

  // Otherwise, the input is '.mlir'.
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
      llvm::MemoryBuffer::getFileOrSTDIN(inputFilename);
  if (std::error_code ec = fileOrErr.getError()) {
    llvm::errs() << "Could not open input file: " << ec.message() << "\n";
    return -1;
  }

  // Parse the input mlir.
  sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
  module = mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);
  if (!module) {
    llvm::errs() << "Error can't load file " << inputFilename << "\n";
    return 3;
  }
  return 0;
}

int dumpMLIR() {
  // llvm::outs()<<configFileName<<"\n";
  initConfig(configFileName);

  diverse = isDiverse;
  std::cout << "d: " << diverse << std::endl;
  diversity.import(dFileName);
  mlir::MLIRContext context;
  // Load our Dialect in this MLIR Context.
  context.getOrLoadDialect<mlir::toy::ToyDialect>();
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
  mlir::SourceMgrDiagnosticHandler sourceMgrHandler(sourceMgr, &context);
  if (int error = loadMLIR(sourceMgr, context, module))
    return error;

  mlir::PassManager pm(module.get()->getName());
  // Apply any generic pass manager command line options and run the pipeline.
  applyPassManagerCLOptions(pm);

  // Check to see what granularity of MLIR we are compiling to.
  bool isLoweringToAffine = emitAction >= Action::DumpMLIRAffine;

  // if (enableOpt || isLoweringToAffine) {
  //   pm.addPass(mlir::createInlinerPass());

  //   mlir::OpPassManager &optPM = pm.nest<mlir::toy::FuncOp>();
  //   optPM.addPass(mlir::toy::createShapeInferencePass());
  //   optPM.addPass(mlir::createCSEPass());
  // }

  if (isLoweringToAffine) {
    // Partially lower the toy dialect.
    pm.addPass(createMLIRSmithPass());

    // Add a few cleanups post lowering.
    // mlir::OpPassManager &optPM = pm.nest<mlir::func::FuncOp>();
    // Partially lower the toy dialect with a few cleanups afterwards.
    //    optPM.addPass(mlir::createCanonicalizerPass());
    //    optPM.addPass(mlir::createCSEPass());

    // Add optimizations if enabled.
    // if (enableOpt) {
    //   optPM.addPass(mlir::affine::createLoopFusionPass());
    //   optPM.addPass(mlir::affine::createAffineScalarReplacementPass());
    // }
  }
  if (mlir::failed(pm.run(*module)))
    return 4;
  module->dump();

  diversity.exportSelf("cov.json");
  return 0;
}

int dumpConfig() {
  printConfig();
  return 0;
}

int dumpAST() {
  if (inputType == InputType::MLIR) {
    llvm::errs() << "Can't dump a Toy AST when the input is MLIR\n";
    return 5;
  }

  auto moduleAST = parseInputFile(inputFilename);
  if (!moduleAST)
    return 1;

  dump(*moduleAST);
  return 0;
}

int main(int argc, char **argv) {
  // Register any command line options.
  mlir::registerAsmPrinterCLOptions();
  mlir::registerMLIRContextCLOptions();
  mlir::registerPassManagerCLOptions();
  //  mlir::init
  cl::ParseCommandLineOptions(argc, argv, "toy compiler\n");
  llvm::InitLLVM(argc, argv);

  switch (emitAction) {
  case Action::DumpConfig:
    // llvm::outs()<<"1\n";
    return dumpConfig();
  case Action::DumpAST:
    // llvm::outs() << "2\n";
    return dumpAST();
  case Action::DumpMLIR:
    // llvm::outs() << "3\n";
  case Action::DumpMLIRAffine:
    // llvm::outs() << "4\n";
    return dumpMLIR();
  default:
    llvm::errs() << "No action specified (parsing only?), use -emit=<action>\n";
  }

  return 0;
}
