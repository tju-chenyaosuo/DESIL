#include <cmath>
#include <random>

#include "mlir/InitAllDialects.h"

#include "fixub/ConstantPool.h"
#include "fixub/FixArith.h"
#include "fixub/FixUBUtils.h"

void FixArith::fixarith(mlir::Operation *op) {
  std::string opname = op->getName().getStringRef().str();
  if (opname == "arith.shli" || opname == "arith.shrsi" ||
      opname == "arith.shrui") {
    FixArith::fixshlr(op);
  }
  if (opname == "arith.ceildivsi" || opname == "arith.divsi" ||
      opname == "arith.floordivsi") {
    FixArith::fixdivsi(op);
  }
  if (opname == "arith.ceildivui" || opname == "arith.divui" ||
      opname == "arith.remui") {
    FixArith::fixdivui(op);
  }
  if (opname == "arith.remsi") {
    FixArith::fixremsi(op);
  }
  return;
}

void FixArith::fixshlr(mlir::Operation *op) { // viewed
  /**
   * Fix undefined bahaviors in arith.shli, arith.shrsi and arith.shrui.
   * The description for these operations in documentation is that:
   * If the value of the second operand(the shift bits) is greater or equal than
   * the bitwidth of the first operand, then the operation returns poison.
   */
  // FixUBUtils mp=FixUBUtils();
  // llvm::outs()<<op->getName()<<"\n";
  // 20240222 deleted, due to the #80960 fix, merge the i1 logic with all
  // integer types
  // if (FixUBUtils::getValueTypeStr(op->getOperand(0)) != "i1")
  // {
  // Fix ub: If the value of the second operand is greater or equal than the
  // bitwidth of the first operand, then the operation returns poison. op1: the
  // number, op2: the shift bits.
  mlir::OpBuilder builder(op->getBlock(), op->getBlock()->begin());
  builder.setInsertionPoint(op);
  mlir::Location loc = op->getLoc();

  mlir::Value val1 = op->getOperand(0), val2 = op->getOperand(1);
  uint64_t bitwidth = val1.getType().getIntOrFloatBitWidth();
  auto bitwidthconstop = builder.create<mlir::arith::ConstantOp>(
      op->getLoc(), mlir::IntegerAttr::get(val1.getType(), bitwidth));
  auto comparisonop = builder.create<mlir::arith::CmpIOp>(
      op->getLoc(), mlir::arith::CmpIPredicate::uge, val2,
      bitwidthconstop.getResult());

  auto GreaterBuilder = [&](mlir::OpBuilder &b, mlir::Location loc) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<unsigned> dis(0, bitwidth - 1);
    unsigned randomValue = dis(gen);
    auto rdmconstantop = b.create<mlir::arith::ConstantOp>(
        loc, mlir::IntegerAttr::get(val1.getType(), randomValue));
    b.create<mlir::scf::YieldOp>(loc, rdmconstantop.getResult());
  };
  auto NotGreaterBuilder = [&](mlir::OpBuilder &b, mlir::Location loc) {
    b.create<mlir::scf::YieldOp>(loc, op->getOperand(1));
  };
  auto ifOp =
      builder.create<mlir::scf::IfOp>(op->getLoc(), comparisonop->getResult(0),
                                      GreaterBuilder, NotGreaterBuilder);
  op->setOperand(1, ifOp.getResult(0));
}

void FixArith::fixdivsi(mlir::Operation *op) { // viewed
  /**
   * Fix undefined behaviors in arith.ceildivsi, arith.divsi, and
   * arith.floordivsi The description of ubs is: Divison by zero, or signed
   * division overflow (minimum value divided by -1) is undefined behavior.
   */
  mlir::OpBuilder builder(op->getBlock(), op->getBlock()->begin());
  builder.setInsertionPoint(op);
  mlir::Location loc = op->getLoc();

  mlir::Value value = op->getOperand(1);
  auto minConstant = builder.create<mlir::arith::ConstantOp>(
      op->getLoc(), value.getType(),
      mlir::IntegerAttr::get(
          value.getType(),
          FixUBUtils::bitsmin(
              op->getOperand(0).getType().getIntOrFloatBitWidth())));
  builder.setInsertionPoint(op);
  auto comparison1op = builder.create<mlir::arith::CmpIOp>(
      op->getLoc(), mlir::arith::CmpIPredicate::eq, op->getOperand(0),
      minConstant.getResult()); // whether 1 is int_min

  int64_t bitw1 = op->getOperand(0).getType().getIntOrFloatBitWidth();
  mlir::arith::AndIOp andop; // prefer to use "invalid" as variable name...
  if (bitw1 == 1) {
    auto falseop = builder.create<mlir::arith::ConstantOp>(
        op->getLoc(), value.getType(),
        mlir::IntegerAttr::get(value.getType(), 0));
    andop = builder.create<mlir::arith::AndIOp>(
        op->getLoc(), comparison1op.getResult(), falseop.getResult());
  } else {
    auto minus1op = builder.create<mlir::arith::ConstantOp>(
        op->getLoc(), value.getType(),
        mlir::IntegerAttr::get(value.getType(), -1));
    auto comparison2op = builder.create<mlir::arith::CmpIOp>(
        op->getLoc(), mlir::arith::CmpIPredicate::eq, op->getOperand(1),
        minus1op.getResult());
    andop = builder.create<mlir::arith::AndIOp>(
        op->getLoc(), comparison1op.getResult(), comparison2op.getResult());
  }

  auto EqminBuilder = [&](mlir::OpBuilder &b, mlir::Location loc) {
    // 生成随机非0,-1数
    int64_t bitwidth = op->getOperand(1).getType().getIntOrFloatBitWidth();
    int64_t rdmnum = 0;
    if (bitwidth == 1)
      rdmnum = 1;
    else
      while (rdmnum == 0 || rdmnum == -1)
        rdmnum = FixUBUtils::randsval(bitwidth);
    auto rdmval = b.create<mlir::arith::ConstantOp>(
                       op->getLoc(), value.getType(),
                       mlir::IntegerAttr::get(value.getType(), rdmnum))
                      .getResult();
    b.create<mlir::scf::YieldOp>(loc, rdmval);
  };
  auto NotEqminBuilder = [&](mlir::OpBuilder &b, mlir::Location loc) {
    auto zeroconstantop = builder.create<mlir::arith::ConstantOp>(
        loc, value.getType(), mlir::IntegerAttr::get(value.getType(), 0));
    auto comparison3op = builder.create<mlir::arith::CmpIOp>(
        loc, mlir::arith::CmpIPredicate::eq, op->getOperand(1),
        zeroconstantop.getResult());
    auto Eq0Builder = [&](mlir::OpBuilder &insideb, mlir::Location insideloc) {
      int64_t bitwidth = op->getOperand(1).getType().getIntOrFloatBitWidth();
      int64_t rdmnum = 0;
      if (bitwidth == 1)
        rdmnum = 1;
      else
        while (rdmnum == 0)
          rdmnum = FixUBUtils::randsval(bitwidth);
      auto rdmval = insideb
                        .create<mlir::arith::ConstantOp>(
                            insideloc, value.getType(),
                            mlir::IntegerAttr::get(value.getType(), rdmnum))
                        .getResult();
      b.create<mlir::scf::YieldOp>(loc, rdmval);
    };
    auto NotEq0Builder = [&](mlir::OpBuilder &insideb,
                             mlir::Location insideloc) {
      b.create<mlir::scf::YieldOp>(loc, op->getOperand(1));
    };
    auto insideifOp = builder.create<mlir::scf::IfOp>(
        loc, comparison3op->getResult(0), Eq0Builder, NotEq0Builder);
    b.create<mlir::scf::YieldOp>(loc, insideifOp.getResult(0));
  };
  auto ifOp = builder.create<mlir::scf::IfOp>(op->getLoc(), andop->getResult(0),
                                              EqminBuilder, NotEqminBuilder);
  op->setOperand(1, ifOp.getResult(0));
}

void FixArith::fixdivui(mlir::Operation *op) { // viewed
  mlir::OpBuilder builder(op->getBlock(), op->getBlock()->begin());
  builder.setInsertionPoint(op);
  mlir::Location loc = op->getLoc();

  mlir::Value value = op->getOperand(1);
  auto zeroConstant = builder.create<mlir::arith::ConstantOp>(
      op->getLoc(), value.getType(),
      mlir::IntegerAttr::get(value.getType(), 0));
  builder.setInsertionPoint(op);
  auto comparisonop = builder.create<mlir::arith::CmpIOp>(
      op->getLoc(), mlir::arith::CmpIPredicate::eq, op->getOperand(1),
      zeroConstant.getResult());

  auto Eq0Builder = [&](mlir::OpBuilder &b, mlir::Location loc) {
    uint64_t bitwidth = op->getOperand(1).getType().getIntOrFloatBitWidth();
    uint64_t rdmnum = 0;
    if (bitwidth == 1)
      rdmnum = 1;
    else
      while (rdmnum == 0)
        rdmnum = FixUBUtils::randuval(bitwidth);
    auto rdmval = b.create<mlir::arith::ConstantOp>(
                       op->getLoc(), value.getType(),
                       mlir::IntegerAttr::get(value.getType(), rdmnum))
                      .getResult();
    b.create<mlir::scf::YieldOp>(loc, rdmval);
  };
  auto NotEq0Builder = [&](mlir::OpBuilder &b, mlir::Location loc) {
    b.create<mlir::scf::YieldOp>(loc, op->getOperand(1));
  };
  auto ifOp = builder.create<mlir::scf::IfOp>(
      op->getLoc(), comparisonop->getResult(0), Eq0Builder, NotEq0Builder);
  op->setOperand(1, ifOp.getResult(0));
}

void FixArith::fixremsi(mlir::Operation *op) { // viewed
  mlir::OpBuilder builder(op->getBlock(), op->getBlock()->begin());
  builder.setInsertionPoint(op);
  mlir::Location loc = op->getLoc();

  mlir::Value value = op->getOperand(1);
  auto zeroConstant = builder.create<mlir::arith::ConstantOp>(
      op->getLoc(), value.getType(),
      mlir::IntegerAttr::get(value.getType(), 0));
  builder.setInsertionPoint(op);
  auto comparisonop = builder.create<mlir::arith::CmpIOp>(
      op->getLoc(), mlir::arith::CmpIPredicate::eq, op->getOperand(1),
      zeroConstant.getResult());

  auto Eq0Builder = [&](mlir::OpBuilder &b, mlir::Location loc) {
    int64_t bitwidth = op->getOperand(1).getType().getIntOrFloatBitWidth();
    int64_t rdmnum = 0;
    if (bitwidth == 1)
      rdmnum = 1;
    else
      while (rdmnum == 0)
        rdmnum = FixUBUtils::randsval(bitwidth);
    auto rdmval = b.create<mlir::arith::ConstantOp>(
                       op->getLoc(), value.getType(),
                       mlir::IntegerAttr::get(value.getType(), rdmnum))
                      .getResult();
    b.create<mlir::scf::YieldOp>(loc, rdmval);
  };
  auto NotEq0Builder = [&](mlir::OpBuilder &b, mlir::Location loc) {
    b.create<mlir::scf::YieldOp>(loc, op->getOperand(1));
  };
  auto ifOp = builder.create<mlir::scf::IfOp>(
      op->getLoc(), comparisonop->getResult(0), Eq0Builder, NotEq0Builder);
  op->setOperand(1, ifOp.getResult(0));
}