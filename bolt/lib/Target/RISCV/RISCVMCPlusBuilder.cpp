//===- bolt/Target/RISCV/RISCVMCPlusBuilder.cpp -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides RISCV-specific MCPlus builder.
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/RISCVMCExpr.h"
#include "MCTargetDesc/RISCVMCTargetDesc.h"
#include "bolt/Core/MCPlusBuilder.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstBuilder.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "mcplus"

using namespace llvm;
using namespace bolt;

namespace {

class RISCVMCPlusBuilder : public MCPlusBuilder {
public:
  using MCPlusBuilder::MCPlusBuilder;

  bool shouldRecordCodeRelocation(uint64_t RelType) const override {
    switch (RelType) {
    case ELF::R_RISCV_JAL:
    case ELF::R_RISCV_CALL:
    case ELF::R_RISCV_CALL_PLT:
    case ELF::R_RISCV_BRANCH:
    case ELF::R_RISCV_RVC_BRANCH:
    case ELF::R_RISCV_RVC_JUMP:
    case ELF::R_RISCV_GOT_HI20:
    case ELF::R_RISCV_PCREL_HI20:
    case ELF::R_RISCV_PCREL_LO12_I:
    case ELF::R_RISCV_PCREL_LO12_S:
      return true;
    default:
      llvm_unreachable("Unexpected RISCV relocation type in code");
    }
  }

  bool isNop(const MCInst &Inst) const {
    return Inst.getOpcode() == RISCV::ADDI &&
           Inst.getOperand(0).getReg() == RISCV::X0 &&
           Inst.getOperand(1).getReg() == RISCV::X0 &&
           Inst.getOperand(2).getImm() == 0;
  }

  bool isCNop(const MCInst &Inst) const {
    return Inst.getOpcode() == RISCV::C_NOP;
  }

  bool isNoop(const MCInst &Inst) const override {
    return isNop(Inst) || isCNop(Inst);
  }

  bool isPseudo(const MCInst &Inst) const override {
    switch (Inst.getOpcode()) {
    default:
      return MCPlusBuilder::isPseudo(Inst);
    case RISCV::PseudoCALL:
    case RISCV::PseudoTAIL:
      return false;
    }
  }

  bool isIndirectCall(const MCInst &Inst) const override {
    if (!isCall(Inst))
      return false;

    switch (Inst.getOpcode()) {
    default:
      return false;
    case RISCV::JALR:
    case RISCV::C_JALR:
      return true;
    }
  }

  bool hasPCRelOperand(const MCInst &Inst) const override {
    switch (Inst.getOpcode()) {
    default:
      return false;
    case RISCV::JAL:
    case RISCV::AUIPC:
      return true;
    }
  }

  unsigned getInvertedBranchOpcode(unsigned Opcode) const {
    switch (Opcode) {
    default:
      llvm_unreachable("Failed to invert branch opcode");
      return Opcode;
    case RISCV::BEQ:
      return RISCV::BNE;
    case RISCV::BNE:
      return RISCV::BEQ;
    case RISCV::BLT:
      return RISCV::BGE;
    case RISCV::BGE:
      return RISCV::BLT;
    case RISCV::BLTU:
      return RISCV::BGEU;
    case RISCV::BGEU:
      return RISCV::BLTU;
    case RISCV::C_BEQZ:
      return RISCV::C_BNEZ;
    case RISCV::C_BNEZ:
      return RISCV::C_BEQZ;
    }
  }

  bool reverseBranchCondition(MCInst &Inst, const MCSymbol *TBB,
                              MCContext *Ctx) const override {
    auto Opcode = getInvertedBranchOpcode(Inst.getOpcode());
    Inst.setOpcode(Opcode);
    return replaceBranchTarget(Inst, TBB, Ctx);
  }

  bool replaceBranchTarget(MCInst &Inst, const MCSymbol *TBB,
                           MCContext *Ctx) const override {
    assert((isCall(Inst) || isBranch(Inst)) && !isIndirectBranch(Inst) &&
           "Invalid instruction");

    unsigned SymOpIndex;
    auto Result = getSymbolRefOperandNum(Inst, SymOpIndex);
    (void)Result;
    assert(Result && "unimplemented branch");

    Inst.getOperand(SymOpIndex) = MCOperand::createExpr(
        MCSymbolRefExpr::create(TBB, MCSymbolRefExpr::VK_None, *Ctx));
    return true;
  }

  IndirectBranchType analyzeIndirectBranch(
      MCInst &Instruction, InstructionIterator Begin, InstructionIterator End,
      const unsigned PtrSize, MCInst *&MemLocInstr, unsigned &BaseRegNum,
      unsigned &IndexRegNum, int64_t &DispValue, const MCExpr *&DispExpr,
      MCInst *&PCRelBaseOut) const override {
    MemLocInstr = nullptr;
    BaseRegNum = 0;
    IndexRegNum = 0;
    DispValue = 0;
    DispExpr = nullptr;
    PCRelBaseOut = nullptr;
    return IndirectBranchType::UNKNOWN;
  }

  bool convertJmpToTailCall(MCInst &Inst) override {
    if (isTailCall(Inst))
      return false;

    switch (Inst.getOpcode()) {
    default:
      llvm_unreachable("unsupported tail call opcode");
    case RISCV::JAL:
    case RISCV::JALR:
    case RISCV::C_J:
    case RISCV::C_JR:
      break;
    }

    setTailCall(Inst);
    return true;
  }

  bool createReturn(MCInst &Inst) const override {
    // TODO "c.jr ra" when RVC is enabled
    Inst.setOpcode(RISCV::JALR);
    Inst.clear();
    Inst.addOperand(MCOperand::createReg(RISCV::X0));
    Inst.addOperand(MCOperand::createReg(RISCV::X1));
    Inst.addOperand(MCOperand::createImm(0));
    return true;
  }

  bool createUncondBranch(MCInst &Inst, const MCSymbol *TBB,
                          MCContext *Ctx) const override {
    Inst.setOpcode(RISCV::JAL);
    Inst.clear();
    Inst.addOperand(MCOperand::createReg(RISCV::X0));
    Inst.addOperand(MCOperand::createExpr(
        MCSymbolRefExpr::create(TBB, MCSymbolRefExpr::VK_None, *Ctx)));
    return true;
  }

  StringRef getTrapFillValue() const override {
    return StringRef("\0\0\0\0", 4);
  }

  bool createCall(unsigned Opcode, MCInst &Inst, const MCSymbol *Target,
                  MCContext *Ctx) {
    Inst.setOpcode(Opcode);
    Inst.clear();
    Inst.addOperand(MCOperand::createExpr(RISCVMCExpr::create(
        MCSymbolRefExpr::create(Target, MCSymbolRefExpr::VK_None, *Ctx),
        RISCVMCExpr::VK_RISCV_CALL, *Ctx)));
    return true;
  }

  bool createCall(MCInst &Inst, const MCSymbol *Target,
                  MCContext *Ctx) override {
    return createCall(RISCV::PseudoCALL, Inst, Target, Ctx);
  }

  bool createTailCall(MCInst &Inst, const MCSymbol *Target,
                      MCContext *Ctx) override {
    return createCall(RISCV::PseudoTAIL, Inst, Target, Ctx);
  }

  bool analyzeBranch(InstructionIterator Begin, InstructionIterator End,
                     const MCSymbol *&TBB, const MCSymbol *&FBB,
                     MCInst *&CondBranch,
                     MCInst *&UncondBranch) const override {
    auto I = End;

    while (I != Begin) {
      --I;

      // Ignore nops and CFIs
      if (isPseudo(*I) || isNoop(*I))
        continue;

      // Stop when we find the first non-terminator
      if (!isTerminator(*I) || isTailCall(*I) || !isBranch(*I))
        break;

      // Handle unconditional branches.
      if (isUnconditionalBranch(*I)) {
        // If any code was seen after this unconditional branch, we've seen
        // unreachable code. Ignore them.
        CondBranch = nullptr;
        UncondBranch = &*I;
        const MCSymbol *Sym = getTargetSymbol(*I);
        assert(Sym != nullptr &&
               "Couldn't extract BB symbol from jump operand");
        TBB = Sym;
        continue;
      }

      // Handle conditional branches and ignore indirect branches
      if (isIndirectBranch(*I))
        return false;

      if (CondBranch == nullptr) {
        const MCSymbol *TargetBB = getTargetSymbol(*I);
        if (TargetBB == nullptr) {
          // Unrecognized branch target
          return false;
        }
        FBB = TBB;
        TBB = TargetBB;
        CondBranch = &*I;
        continue;
      }

      llvm_unreachable("multiple conditional branches in one BB");
    }

    return true;
  }

  bool getSymbolRefOperandNum(const MCInst &Inst, unsigned &OpNum) const {
    switch (Inst.getOpcode()) {
    default:
      return false;
    case RISCV::C_J:
    case RISCV::PseudoCALL:
    case RISCV::PseudoTAIL:
      OpNum = 0;
      return true;
    case RISCV::AUIPC:
    case RISCV::JAL:
    case RISCV::C_BEQZ:
    case RISCV::C_BNEZ:
      OpNum = 1;
      return true;
    case RISCV::BEQ:
    case RISCV::BGE:
    case RISCV::BGEU:
    case RISCV::BNE:
    case RISCV::BLT:
    case RISCV::BLTU:
      OpNum = 2;
      return true;
    }
  }

  const MCSymbol *getTargetSymbol(const MCExpr *Expr) const override {
    auto *RISCVExpr = dyn_cast<RISCVMCExpr>(Expr);
    if (RISCVExpr && RISCVExpr->getSubExpr())
      return getTargetSymbol(RISCVExpr->getSubExpr());

    auto *BinExpr = dyn_cast<MCBinaryExpr>(Expr);
    if (BinExpr)
      return getTargetSymbol(BinExpr->getLHS());

    auto *SymExpr = dyn_cast<MCSymbolRefExpr>(Expr);
    if (SymExpr && SymExpr->getKind() == MCSymbolRefExpr::VK_None)
      return &SymExpr->getSymbol();

    return nullptr;
  }

  const MCSymbol *getTargetSymbol(const MCInst &Inst,
                                  unsigned OpNum = 0) const override {
    if (!OpNum && !getSymbolRefOperandNum(Inst, OpNum))
      return nullptr;

    const MCOperand &Op = Inst.getOperand(OpNum);
    if (!Op.isExpr())
      return nullptr;

    return getTargetSymbol(Op.getExpr());
  }

  bool lowerTailCall(MCInst &Inst) override {
    removeAnnotation(Inst, MCPlus::MCAnnotation::kTailCall);
    if (getConditionalTailCall(Inst))
      unsetConditionalTailCall(Inst);
    return true;
  }

  uint64_t analyzePLTEntry(MCInst &Instruction, InstructionIterator Begin,
                           InstructionIterator End,
                           uint64_t BeginPC) const override {
    auto I = Begin;

    assert(I != End);
    auto &AUIPC = *I++;
    assert(AUIPC.getOpcode() == RISCV::AUIPC);
    assert(AUIPC.getOperand(0).getReg() == RISCV::X28);

    assert(I != End);
    auto &LD = *I++;
    assert(LD.getOpcode() == RISCV::LD);
    assert(LD.getOperand(0).getReg() == RISCV::X28);
    assert(LD.getOperand(1).getReg() == RISCV::X28);

    assert(I != End);
    auto &JALR = *I++;
    (void)JALR;
    assert(JALR.getOpcode() == RISCV::JALR);
    assert(JALR.getOperand(0).getReg() == RISCV::X6);
    assert(JALR.getOperand(1).getReg() == RISCV::X28);

    assert(I != End);
    auto &NOP = *I++;
    (void)NOP;
    assert(isNoop(NOP));

    assert(I == End);

    auto AUIPCOffset = AUIPC.getOperand(1).getImm() << 12;
    auto LDOffset = LD.getOperand(2).getImm();
    return BeginPC + AUIPCOffset + LDOffset;
  }

  bool replaceImmWithSymbolRef(MCInst &Inst, const MCSymbol *Symbol,
                               int64_t Addend, MCContext *Ctx, int64_t &Value,
                               uint64_t RelType) const override {
    unsigned ImmOpNo = -1U;

    for (unsigned Index = 0; Index < MCPlus::getNumPrimeOperands(Inst);
         ++Index) {
      if (Inst.getOperand(Index).isImm()) {
        ImmOpNo = Index;
        break;
      }
    }

    if (ImmOpNo == -1U)
      return false;

    Value = Inst.getOperand(ImmOpNo).getImm();
    setOperandToSymbolRef(Inst, ImmOpNo, Symbol, Addend, Ctx, RelType);
    return true;
  }

  const MCExpr *getTargetExprFor(MCInst &Inst, const MCExpr *Expr,
                                 MCContext &Ctx,
                                 uint64_t RelType) const override {
    switch (RelType) {
    default:
      return Expr;
    case ELF::R_RISCV_GOT_HI20:
      // The GOT is reused so no need to create GOT relocations
    case ELF::R_RISCV_PCREL_HI20:
      return RISCVMCExpr::create(Expr, RISCVMCExpr::VK_RISCV_PCREL_HI, Ctx);
    case ELF::R_RISCV_PCREL_LO12_I:
    case ELF::R_RISCV_PCREL_LO12_S:
      return RISCVMCExpr::create(Expr, RISCVMCExpr::VK_RISCV_PCREL_LO, Ctx);
    case ELF::R_RISCV_CALL:
      return RISCVMCExpr::create(Expr, RISCVMCExpr::VK_RISCV_CALL, Ctx);
    case ELF::R_RISCV_CALL_PLT:
      return RISCVMCExpr::create(Expr, RISCVMCExpr::VK_RISCV_CALL_PLT, Ctx);
    }
  }

  bool evaluateMemOperandTarget(const MCInst &Inst, uint64_t &Target,
                                uint64_t Address,
                                uint64_t Size) const override {
    return false;
  }

  bool isCallAuipc(const MCInst &Inst) const {
    if (Inst.getOpcode() != RISCV::AUIPC)
      return false;

    const auto &ImmOp = Inst.getOperand(1);
    if (!ImmOp.isExpr())
      return false;

    const auto *ImmExpr = ImmOp.getExpr();
    if (!isa<RISCVMCExpr>(ImmExpr))
      return false;

    switch (cast<RISCVMCExpr>(ImmExpr)->getKind()) {
    default:
      return false;
    case RISCVMCExpr::VK_RISCV_CALL:
    case RISCVMCExpr::VK_RISCV_CALL_PLT:
      return true;
    }
  }

  bool isRISCVCall(const MCInst &First, const MCInst &Second) const override {
    if (!isCallAuipc(First))
      return false;

    assert(Second.getOpcode() == RISCV::JALR);
    return true;
  }

  bool createMarker(MCInst &Inst, int I) const {
    Inst.clear();
    Inst.setOpcode(RISCV::ADDI);
    Inst.addOperand(MCOperand::createReg(RISCV::X0));
    Inst.addOperand(MCOperand::createReg(RISCV::X0));
    Inst.addOperand(MCOperand::createImm(I));
    return true;
  }

  void createLI(InstructionListType &Insts, unsigned Reg, int64_t Imm) {
    assert(isInt<12>(Imm) && "not supported");
    auto ADDI =
        MCInstBuilder(RISCV::ADDI).addReg(Reg).addReg(RISCV::X0).addImm(Imm);
    Insts.push_back(ADDI);
  }

  void replaceIndJumpWithLoadAddress(MCInst &Jump, unsigned DestReg) {
    switch (Jump.getOpcode()) {
    default:
      llvm_unreachable("unsupported indirect jump");
    case RISCV::JALR:
      Jump.setOpcode(RISCV::ADDI);
      Jump.getOperand(0).setReg(DestReg);
      break;
    case RISCV::C_JALR:
    case RISCV::C_JR: {
      auto TargetReg = Jump.getOperand(0).getReg();
      Jump = MCInstBuilder(RISCV::ADDI)
                 .addReg(DestReg)
                 .addReg(TargetReg)
                 .addImm(0);
    }
    }
  }

  // InstructionListType
  // createInstrumentedIndirectCall(MCInst &&CallInst, MCSymbol *HandlerFuncAddr,
  //                                int CallSiteID, MCContext *Ctx) override {
  //   InstructionListType Insts;
  //   createMarker(Insts.emplace_back(), 11);

  //   auto IsTailCall = isTailCall(CallInst);
  //   createStackSaveArea(Insts.emplace_back(), IsTailCall ? 3 : 2);

  //   if (IsTailCall)
  //     createRegStackSave(Insts.emplace_back(), RISCV::X1, 2);

  //   // Compute target: replace jalr with addi and rd with t0
  //   replaceIndJumpWithLoadAddress(CallInst, RISCV::X5);
  //   stripAnnotations(CallInst);
  //   Insts.push_back(CallInst);
  //   createRegStackSave(Insts.emplace_back(), RISCV::X5, 1);

  //   // Store CallSiteID on the stack.
  //   createLI(Insts, RISCV::X5, CallSiteID);
  //   createRegStackSave(Insts.emplace_back(), RISCV::X5, 0);

  //   createCall(Insts.emplace_back(), HandlerFuncAddr, Ctx);

  //   return Insts;
  // }

  InstructionListType
  createInstrumentedIndirectCall(MCInst &&CallInst, MCSymbol *HandlerFuncAddr,
                                 int CallSiteID, MCContext *Ctx) override {
    InstructionListType Insts;
    createMarker(Insts.emplace_back(), 11);

    auto IsTailCall = isTailCall(CallInst);

    // Compute target: replace jalr with addi and rd with t0
    replaceIndJumpWithLoadAddress(CallInst, RISCV::X30);
    stripAnnotations(CallInst);
    Insts.push_back(CallInst);

    // Store CallSiteID on the stack.
    createLI(Insts, RISCV::X31, CallSiteID);

    if (IsTailCall)
      createTailCall(Insts.emplace_back(), HandlerFuncAddr, Ctx);
    else
      createCall(Insts.emplace_back(), HandlerFuncAddr, Ctx);

    return Insts;
  }

  void createLoad(InstructionListType &Insts, unsigned DestReg,
                const MCSymbol *Target, MCContext *Ctx) {
    auto &AUIPC = Insts.emplace_back();
    AUIPC.setOpcode(RISCV::AUIPC);
    AUIPC.addOperand(MCOperand::createReg(DestReg));
    AUIPC.addOperand(MCOperand::createExpr(
        RISCVMCExpr::create(MCSymbolRefExpr::create(Target, *Ctx),
                            RISCVMCExpr::VK_RISCV_PCREL_HI, *Ctx)));
    auto *AUIPCLabel = Ctx->createNamedTempSymbol();
    setLabel(AUIPC, AUIPCLabel);

    auto &LD = Insts.emplace_back();
    LD.setOpcode(RISCV::LD);
    LD.addOperand(MCOperand::createReg(DestReg));
    LD.addOperand(MCOperand::createReg(DestReg));
    LD.addOperand(MCOperand::createExpr(
        RISCVMCExpr::create(MCSymbolRefExpr::create(AUIPCLabel, *Ctx),
                            RISCVMCExpr::VK_RISCV_PCREL_LO, *Ctx)));
  }

  void createBranch(MCInst &Inst, unsigned Opcode, unsigned RS1, unsigned RS2,
                    const MCSymbol *Target, MCContext *Ctx) {
    Inst = MCInstBuilder(Opcode).addReg(RS1).addReg(RS2).addExpr(
        MCSymbolRefExpr::create(Target, *Ctx));
  }

  void createBEQZ(MCInst &Inst, unsigned Reg, const MCSymbol *Target,
                  MCContext *Ctx) {
    createBranch(Inst, RISCV::BEQ, Reg, RISCV::X0, Target, Ctx);
  }

  void createIndirectCall(MCInst &Inst, unsigned Reg, int Offset = 0) const {
    Inst =
        MCInstBuilder(RISCV::JALR).addReg(RISCV::X1).addReg(Reg).addImm(Offset);
  }

  void createIndirectJump(MCInst &Inst, unsigned Reg, int Offset = 0) const {
    Inst =
        MCInstBuilder(RISCV::JALR).addReg(RISCV::X0).addReg(Reg).addImm(Offset);
  }

  void createMV(MCInst &Inst, unsigned Dest, unsigned Src) const {
    Inst = MCInstBuilder(RISCV::ADDI).addReg(Dest).addReg(Src).addImm(0);
  }

  InstructionListType
  createInstrumentedIndCallHandlerEntryBB(const MCSymbol *InstrTrampoline,
                                          const MCSymbol *IndCallHandler,
                                          MCContext *Ctx) override {
    InstructionListType Insts;
    createMarker(Insts.emplace_back(), 1);
    createMV(Insts.emplace_back(), RISCV::X29, RISCV::X1);
    createLoad(Insts, RISCV::X5, InstrTrampoline, Ctx);
    createBEQZ(Insts.emplace_back(), RISCV::X5, IndCallHandler, Ctx);
    createIndirectCall(Insts.emplace_back(), RISCV::X5);
    createCall(RISCV::PseudoTAIL, Insts.emplace_back(), IndCallHandler, Ctx);
    return Insts;
  }

  // InstructionListType createInstrumentedIndCallHandlerExitBB() const override {
  //   InstructionListType Insts;
  //   createRegStackRestore(Insts.emplace_back(), RISCV::X5, 1);
  //   restoreStackSaveArea(Insts.emplace_back(), 2);
  //   createIndirectJump(Insts.emplace_back(), RISCV::X5);
  //   return Insts;
  // }

  InstructionListType createInstrumentedIndCallHandlerExitBB() const override {
    InstructionListType Insts;
    createMV(Insts.emplace_back(), RISCV::X1, RISCV::X29);
    createIndirectJump(Insts.emplace_back(), RISCV::X30);
    return Insts;
  }

  // InstructionListType
  // createInstrumentedIndTailCallHandlerExitBB() const override {
  //   InstructionListType Insts;
  //   createRegStackRestore(Insts.emplace_back(), RISCV::X1, 2);
  //   createRegStackRestore(Insts.emplace_back(), RISCV::X5, 1);
  //   restoreStackSaveArea(Insts.emplace_back(), 3);
  //   createIndirectJump(Insts.emplace_back(), RISCV::X5);
  //   return Insts;
  // }

  InstructionListType
  createInstrumentedIndTailCallHandlerExitBB() const override {
    InstructionListType Insts;
    createMV(Insts.emplace_back(), RISCV::X1, RISCV::X29);
    createIndirectJump(Insts.emplace_back(), RISCV::X30);
    return Insts;
  }

  InstructionListType createNumCountersGetter(MCContext *Ctx) const override {
    InstructionListType Insts;
    createMarker(Insts.emplace_back(), 4);
    return Insts;
  }

  InstructionListType
  createInstrLocationsGetter(MCContext *Ctx) const override {
    InstructionListType Insts;
    createMarker(Insts.emplace_back(), 5);
    return Insts;
  }

  InstructionListType createInstrTablesGetter(MCContext *Ctx) const override {
    InstructionListType Insts;
    createMarker(Insts.emplace_back(), 6);
    return Insts;
  }

  InstructionListType createInstrNumFuncsGetter(MCContext *Ctx) const override {
    InstructionListType Insts;
    createMarker(Insts.emplace_back(), 7);
    return Insts;
  }

  InstructionListType createSymbolTrampoline(const MCSymbol *TgtSym,
                                             MCContext *Ctx) override {
    InstructionListType Insts;
    createTailCall(Insts.emplace_back(), TgtSym, Ctx);
    return Insts;
  }

  InstructionListType createDummyReturnFunction(MCContext *Ctx) const override {
    InstructionListType Insts;
    createMarker(Insts.emplace_back(), 9);
    return Insts;
  }

  void createRegInc(MCInst &Inst, unsigned Reg, int64_t Imm) const {
    Inst.setOpcode(RISCV::ADDI);
    Inst.addOperand(MCOperand::createReg(Reg));
    Inst.addOperand(MCOperand::createReg(Reg));
    Inst.addOperand(MCOperand::createImm(Imm));
  }

  void createSPInc(MCInst &Inst, int64_t Imm) const {
    createRegInc(Inst, RISCV::X2, Imm);
  }

  void createStackSaveArea(MCInst &Inst, unsigned NumRegs) {
    createSPInc(Inst, NumRegs * -8);
  }

  void createRegStackSave(MCInst &Inst, unsigned Reg, unsigned RegOffset) {
    createStore(Inst, Reg, RISCV::X2, RegOffset * 8);
  }

  void createRegStackRestore(MCInst &Inst, unsigned Reg,
                             unsigned RegOffset) const {
    createLoad(Inst, Reg, RISCV::X2, RegOffset * 8);
  }

  void restoreStackSaveArea(MCInst &Inst, unsigned NumRegs) const {
    createSPInc(Inst, NumRegs * 8);
  }

  void createStore(MCInst &Inst, unsigned Reg, unsigned BaseReg,
                   int64_t Offset) const {
    Inst.setOpcode(RISCV::SD);
    Inst.addOperand(MCOperand::createReg(Reg));
    Inst.addOperand(MCOperand::createReg(BaseReg));
    Inst.addOperand(MCOperand::createImm(Offset));
  }

  void createLoad(MCInst &Inst, unsigned Reg, unsigned BaseReg,
                  int64_t Offset) const {
    Inst.setOpcode(RISCV::LD);
    Inst.addOperand(MCOperand::createReg(Reg));
    Inst.addOperand(MCOperand::createReg(BaseReg));
    Inst.addOperand(MCOperand::createImm(Offset));
  }

  void spillRegs(InstructionListType &Insts,
                 const SmallVector<unsigned> &Regs) const {
    createSPInc(Insts.emplace_back(), -Regs.size() * 8);

    int64_t Offset = 0;
    for (auto Reg : Regs) {
      createStore(Insts.emplace_back(), Reg, RISCV::X2, Offset);
      Offset += 8;
    }
  }

  void reloadRegs(InstructionListType &Insts,
                  const SmallVector<unsigned> &Regs) {
    int64_t Offset = 0;
    for (auto Reg : Regs) {
      createLoad(Insts.emplace_back(), Reg, RISCV::X2, Offset);
      Offset += 8;
    }

    createSPInc(Insts.emplace_back(), Regs.size() * 8);
  }

  InstructionListType
  createInstrIncMemory(const MCSymbol *Target, MCContext *Ctx, bool IsLeaf,
                       unsigned CodePointerSize) override {
    InstructionListType Insts;
    createMarker(Insts.emplace_back(), 42);

    spillRegs(Insts, {RISCV::X5, RISCV::X6});

    auto &AUIPC = Insts.emplace_back();
    AUIPC.setOpcode(RISCV::AUIPC);
    AUIPC.addOperand(MCOperand::createReg(RISCV::X5));
    AUIPC.addOperand(MCOperand::createExpr(
        RISCVMCExpr::create(MCSymbolRefExpr::create(Target, *Ctx),
                            RISCVMCExpr::VK_RISCV_PCREL_HI, *Ctx)));
    auto *AUIPCLabel = Ctx->createNamedTempSymbol();
    setLabel(AUIPC, AUIPCLabel);

    auto &LD = Insts.emplace_back();
    LD.setOpcode(RISCV::LD);
    LD.addOperand(MCOperand::createReg(RISCV::X6));
    LD.addOperand(MCOperand::createReg(RISCV::X5));
    LD.addOperand(MCOperand::createExpr(
        RISCVMCExpr::create(MCSymbolRefExpr::create(AUIPCLabel, *Ctx),
                            RISCVMCExpr::VK_RISCV_PCREL_LO, *Ctx)));

    createRegInc(Insts.emplace_back(), RISCV::X6, 1);

    auto &SD = Insts.emplace_back();
    SD.setOpcode(RISCV::SD);
    SD.addOperand(MCOperand::createReg(RISCV::X6));
    SD.addOperand(MCOperand::createReg(RISCV::X5));
    SD.addOperand(MCOperand::createExpr(
        RISCVMCExpr::create(MCSymbolRefExpr::create(AUIPCLabel, *Ctx),
                            RISCVMCExpr::VK_RISCV_PCREL_LO, *Ctx)));

    reloadRegs(Insts, {RISCV::X5, RISCV::X6});

    return Insts;
  }
};

} // end anonymous namespace

namespace llvm {
namespace bolt {

MCPlusBuilder *createRISCVMCPlusBuilder(const MCInstrAnalysis *Analysis,
                                        const MCInstrInfo *Info,
                                        const MCRegisterInfo *RegInfo) {
  return new RISCVMCPlusBuilder(Analysis, Info, RegInfo);
}

} // namespace bolt
} // namespace llvm
