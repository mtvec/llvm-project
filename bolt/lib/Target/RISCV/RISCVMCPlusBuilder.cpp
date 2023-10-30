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

#include "MCTargetDesc/RISCVBaseInfo.h"
#include "MCTargetDesc/RISCVMCExpr.h"
#include "MCTargetDesc/RISCVMCTargetDesc.h"
#include "MCTargetDesc/RISCVMatInt.h"
#include "bolt/Core/MCPlusBuilder.h"
#include "bolt/Utils/CommandLineOpts.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstBuilder.h"
#include "llvm/MC/MCSubtargetInfo.h"
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
  using MCPlusBuilder::createLoad;
  using MCPlusBuilder::MCPlusBuilder;

  bool equals(const MCTargetExpr &A, const MCTargetExpr &B,
              CompFuncTy Comp) const override {
    const auto &RISCVExprA = cast<RISCVMCExpr>(A);
    const auto &RISCVExprB = cast<RISCVMCExpr>(B);
    if (RISCVExprA.getKind() != RISCVExprB.getKind())
      return false;

    return MCPlusBuilder::equals(*RISCVExprA.getSubExpr(),
                                 *RISCVExprB.getSubExpr(), Comp);
  }

  void getCalleeSavedRegs(BitVector &Regs) const override {
    Regs |= getAliases(RISCV::X2);
    Regs |= getAliases(RISCV::X8);
    Regs |= getAliases(RISCV::X9);
    Regs |= getAliases(RISCV::X18);
    Regs |= getAliases(RISCV::X19);
    Regs |= getAliases(RISCV::X20);
    Regs |= getAliases(RISCV::X21);
    Regs |= getAliases(RISCV::X22);
    Regs |= getAliases(RISCV::X23);
    Regs |= getAliases(RISCV::X24);
    Regs |= getAliases(RISCV::X25);
    Regs |= getAliases(RISCV::X26);
    Regs |= getAliases(RISCV::X27);
  }

  int getPCRelEncodingSize(const MCInst &Inst) const override {
    switch (Inst.getOpcode()) {
    default:
      llvm_unreachable("unsupported opcode");
    case RISCV::PseudoCALL:
    case RISCV::PseudoTAIL:
    case RISCV::PseudoJump:
      return 32;
    case RISCV::JAL:
      return 21;
    case RISCV::BEQ:
    case RISCV::BNE:
    case RISCV::BLT:
    case RISCV::BGE:
    case RISCV::BLTU:
    case RISCV::BGEU:
      return 21;
      return 13;
    case RISCV::C_J:
      return 12;
    case RISCV::C_BEQZ:
    case RISCV::C_BNEZ:
      return 9;
    }
  }

  size_t getSize(const MCInst &Inst) const override {
    switch (Inst.getOpcode()) {
    default:
      return Info->get(Inst.getOpcode()).getSize();
    case RISCV::BEQ:
    case RISCV::BNE:
    case RISCV::BLT:
    case RISCV::BGE:
    case RISCV::BLTU:
    case RISCV::BGEU:
      return 8;
    }
  }

  int getShortJmpEncodingSize() const override { return 32; }

  int getUncondBranchEncodingSize() const override { return 32; }

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
    case ELF::R_RISCV_HI20:
    case ELF::R_RISCV_LO12_I:
    case ELF::R_RISCV_LO12_S:
    case ELF::R_RISCV_TLS_GOT_HI20:
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
    case RISCV::PseudoJump:
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
    case RISCV::C_JR:
      return true;
    }
  }

  MCInst uncompress(const MCInst &Inst) const {
    MCInst FullInst;
    if (RISCVRVC::uncompress(FullInst, Inst, *STI))
      return FullInst;
    return Inst;
  }

  bool isEpilogue(const MCInst &Inst) const override {
    MCInst FullInst = uncompress(Inst);
    switch (FullInst.getOpcode()) {
    default:
      return false;
    case RISCV::ADDI:
      return FullInst.getOperand(0).getReg() == RISCV::X2 &&
             FullInst.getOperand(1).getReg() == RISCV::X2 &&
             FullInst.getOperand(2).getImm() > 0;
    }
  }

  MCInst::iterator getMemOperandDisp(MCInst &Inst) const override {
    switch (Inst.getOpcode()) {
    default:
      return Inst.end();
    case RISCV::AUIPC:
      return Inst.begin() + 1;
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

  // auipc a1, 0x5f
  // addi  a1, a1, 1060 # 84d40 <.LJTI650_0>
  // add   a0, a0, a1
  // lw    a0, 0(a0)
  // add   a0, a0, a1
  // jr    a0
  bool
  matchJumpTable(const MCInst &Inst,
                 DenseMap<const MCInst *, SmallVector<MCInst *, 4>> &UDChain,
                 const MCExpr *&JumpTable, int64_t &Offset, int64_t &ScaleValue,
                 MCInst *&PCRelBase) const {
    assert(Inst.getOpcode() == RISCV::JALR || Inst.getOpcode() == RISCV::C_JR);

    if (Inst.getOpcode() == RISCV::JALR &&
        Inst.getOperand(0).getReg() != RISCV::X0)
      return false;

    using UsesVec = SmallVector<MCInst *, 4>;
    UsesVec &UsesRoot = UDChain[&Inst];
    if (UsesRoot.size() == 0 || UsesRoot[0] == nullptr)
      return false;

    const MCInst *DefTgtAdd = UsesRoot[0];
    if (DefTgtAdd->getOpcode() != RISCV::ADD &&
        DefTgtAdd->getOpcode() != RISCV::C_ADD)
      return false;

    UsesVec &UsesTgtAdd = UDChain[DefTgtAdd];
    assert(UsesTgtAdd.size() == 3);
    if (!UsesTgtAdd[1] || !UsesTgtAdd[2]) {
      return false;
    }

    const MCInst *DefBaseAddi = UsesTgtAdd[2];
    if (DefBaseAddi->getOpcode() != RISCV::ADDI &&
        DefBaseAddi->getOpcode() != RISCV::C_ADDI)
      return false;

    DefBaseAddi->dump();

    UsesVec &UsesBaseAddi = UDChain[DefBaseAddi];
    assert(UsesBaseAddi.size() == 2);
    if (!UsesBaseAddi[1]) {
      return false;
    }

    MCInst *DefBaseAuipc = UsesBaseAddi[1];
    if (DefBaseAuipc->getOpcode() != RISCV::AUIPC)
      return false;

    assert(DefBaseAuipc->getOperand(1).isExpr());
    JumpTable = DefBaseAuipc->getOperand(1).getExpr();
    PCRelBase = DefBaseAuipc;
    return true;
  }

  IndirectBranchType analyzeIndirectBranch(
      MCInst &Instruction, InstructionIterator Begin, InstructionIterator End,
      const unsigned PtrSize, MCInst *&MemLocInstr, unsigned &BaseRegNum,
      unsigned &IndexRegNum, int64_t &DispValue, const MCExpr *&DispExpr,
      MCInst *&PCRelBase) const override {
    MemLocInstr = nullptr;
    BaseRegNum = 0;
    IndexRegNum = 0;
    DispValue = 0;
    DispExpr = nullptr;
    PCRelBase = nullptr;

    // Check for the following long tail call sequence:
    // 1: auipc xi, %pcrel_hi(sym)
    // jalr zero, %pcrel_lo(1b)(xi)
    if (Instruction.getOpcode() == RISCV::JALR && Begin != End) {
      MCInst &PrevInst = *std::prev(End);
      if (isRISCVCall(PrevInst, Instruction) &&
          Instruction.getOperand(0).getReg() == RISCV::X0)
        return IndirectBranchType::POSSIBLE_TAIL_CALL;
    }

    DenseMap<const MCInst *, SmallVector<llvm::MCInst *, 4>> UDChain =
        computeLocalUDChain(&Instruction, Begin, End);
    int64_t ScaleValue;
    if (matchJumpTable(Instruction, UDChain, DispExpr, DispValue, ScaleValue,
                       PCRelBase)) {
      MemLocInstr = PCRelBase;
      return IndirectBranchType::POSSIBLE_PIC_JUMP_TABLE;
    }

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
    case RISCV::PseudoJump:
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
    Inst.setOpcode(RISCV::PseudoJump);
    Inst.clear();
    Inst.addOperand(MCOperand::createReg(RISCV::X3));
    Inst.addOperand(MCOperand::createExpr(RISCVMCExpr::create(
        MCSymbolRefExpr::create(TBB, MCSymbolRefExpr::VK_None, *Ctx),
        RISCVMCExpr::VK_RISCV_CALL, *Ctx)));
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
      if (isUnconditionalBranch(*I) && !isIndirectBranch(*I)) {
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
    case TargetOpcode::EH_LABEL:
    case RISCV::PseudoCALL:
    case RISCV::PseudoTAIL:
      OpNum = 0;
      return true;
    case RISCV::AUIPC:
    case RISCV::JAL:
    case RISCV::C_BEQZ:
    case RISCV::C_BNEZ:
    case RISCV::PseudoJump:
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

  std::pair<const MCSymbol *, uint64_t>
  getTargetSymbolInfo(const MCExpr *Expr) const override {
    if (const auto *RISCVExpr = dyn_cast<RISCVMCExpr>(Expr))
      return MCPlusBuilder::getTargetSymbolInfo(RISCVExpr->getSubExpr());
    return MCPlusBuilder::getTargetSymbolInfo(Expr);
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
    case ELF::R_RISCV_TLS_GOT_HI20:
      // The GOT is reused so no need to create GOT relocations
    case ELF::R_RISCV_PCREL_HI20:
    case 0:
      return RISCVMCExpr::create(Expr, RISCVMCExpr::VK_RISCV_PCREL_HI, Ctx);
    case ELF::R_RISCV_PCREL_LO12_I:
    case ELF::R_RISCV_PCREL_LO12_S:
      return RISCVMCExpr::create(Expr, RISCVMCExpr::VK_RISCV_PCREL_LO, Ctx);
    case ELF::R_RISCV_HI20:
      return RISCVMCExpr::create(Expr, RISCVMCExpr::VK_RISCV_HI, Ctx);
    case ELF::R_RISCV_LO12_I:
    case ELF::R_RISCV_LO12_S:
      return RISCVMCExpr::create(Expr, RISCVMCExpr::VK_RISCV_LO, Ctx);
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

  uint16_t getMinFunctionAlignment() const override {
    if (STI->hasFeature(RISCV::FeatureStdExtC) ||
        STI->hasFeature(RISCV::FeatureStdExtZca))
      return 2;
    return 4;
  }

  InstructionListType
  createInstrIncMemory(const MCSymbol *Target, MCContext *Ctx, bool IsLeaf,
                       unsigned CodePointerSize) const override {
    // We need 2 scratch registers: one for the target address (t0/x5), and one
    // for the increment value (t1/x6).
    // addi sp, sp, -16
    // sd t0, 0(sp)
    // sd t1, 8(sp)
    // la t0, target         # 1: auipc t0, %pcrel_hi(target)
    //                       # addi t0, t0, %pcrel_lo(1b)
    // li t1, 1              # addi t1, zero, 1
    // amoadd.d zero, t0, t1
    // ld t0, 0(sp)
    // ld t1, 8(sp)
    // addi sp, sp, 16
    InstructionListType Insts;
    spillRegs(Insts, {RISCV::X5, RISCV::X6});

    createLA(Insts, RISCV::X5, Target, *Ctx);

    MCInst LI = MCInstBuilder(RISCV::ADDI)
                    .addReg(RISCV::X6)
                    .addReg(RISCV::X0)
                    .addImm(1);
    Insts.push_back(LI);

    MCInst AMOADD = MCInstBuilder(RISCV::AMOADD_D)
                        .addReg(RISCV::X0)
                        .addReg(RISCV::X5)
                        .addReg(RISCV::X6);
    Insts.push_back(AMOADD);

    reloadRegs(Insts, {RISCV::X5, RISCV::X6});
    return Insts;
  }

  InstructionListType createInstrumentedIndirectCall(MCInst &&CallInst,
                                                     MCSymbol *HandlerFuncAddr,
                                                     int CallSiteID,
                                                     MCContext *Ctx) override {
    // jalr/c.jalr/c.jr offset(xi) is replaced with:
    // addi t5, xi, offset
    // li t6, CallSiteID
    // call/tail HandlerFuncAddr
    InstructionListType Insts;

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

  InstructionListType
  createInstrumentedIndCallHandlerEntryBB(const MCSymbol *InstrTrampoline,
                                          const MCSymbol *IndCallHandler,
                                          MCContext *Ctx) override {
    // mv t4, ra
    // ld t0, InstrTrampoline
    // beqz t0, IndCallHandler
    // jalr t0
    // tail IndCallHandler
    InstructionListType Insts;
    createMV(Insts.emplace_back(), RISCV::X29, RISCV::X1);
    createLoad(Insts, RISCV::X5, InstrTrampoline, *Ctx);
    createBEQZ(Insts.emplace_back(), RISCV::X5, IndCallHandler, Ctx);
    createIndirectCall(Insts.emplace_back(), RISCV::X5);
    createCall(RISCV::PseudoTAIL, Insts.emplace_back(), IndCallHandler, Ctx);
    return Insts;
  }

  InstructionListType createInstrumentedIndCallHandlerExitBB() const override {
    // mv ra, t4
    // jr t5
    InstructionListType Insts;
    createMV(Insts.emplace_back(), RISCV::X1, RISCV::X29);
    createIndirectJump(Insts.emplace_back(), RISCV::X30);
    return Insts;
  }

  InstructionListType
  createInstrumentedIndTailCallHandlerExitBB() const override {
    return createInstrumentedIndCallHandlerExitBB();
  }

  InstructionListType createNumCountersGetter(MCContext *Ctx) const override {
    return {};
  }

  InstructionListType
  createInstrLocationsGetter(MCContext *Ctx) const override {
    return {};
  }

  InstructionListType createInstrTablesGetter(MCContext *Ctx) const override {
    return {};
  }

  InstructionListType createInstrNumFuncsGetter(MCContext *Ctx) const override {
    return {};
  }

  InstructionListType createSymbolTrampoline(const MCSymbol *TgtSym,
                                             MCContext *Ctx) override {
    InstructionListType Insts;
    createTailCall(Insts.emplace_back(), TgtSym, Ctx);
    return Insts;
  }

  InstructionListType createDummyReturnFunction(MCContext *Ctx) const override {
    return {};
  }

  const RISCVMCExpr *createSymbolRefExpr(const MCSymbol *Target,
                                         RISCVMCExpr::VariantKind VK,
                                         MCContext &Ctx) const {
    return RISCVMCExpr::create(MCSymbolRefExpr::create(Target, Ctx), VK, Ctx);
  }

  void createAuipcInstPair(InstructionListType &Insts, unsigned DestReg,
                           const MCSymbol *Target, unsigned SecondOpcode,
                           MCContext &Ctx) const {
    MCInst AUIPC = MCInstBuilder(RISCV::AUIPC)
                       .addReg(DestReg)
                       .addExpr(createSymbolRefExpr(
                           Target, RISCVMCExpr::VK_RISCV_PCREL_HI, Ctx));
    MCSymbol *AUIPCLabel = Ctx.createNamedTempSymbol("pcrel_hi");
    setLabel(AUIPC, AUIPCLabel);
    Insts.push_back(AUIPC);

    MCInst SecondInst =
        MCInstBuilder(SecondOpcode)
            .addReg(DestReg)
            .addReg(DestReg)
            .addExpr(createSymbolRefExpr(AUIPCLabel,
                                         RISCVMCExpr::VK_RISCV_PCREL_LO, Ctx));
    Insts.push_back(SecondInst);
  }

  void createLA(InstructionListType &Insts, unsigned DestReg,
                const MCSymbol *Target, MCContext &Ctx) const {
    createAuipcInstPair(Insts, DestReg, Target, RISCV::ADDI, Ctx);
  }

  void createLoad(InstructionListType &Insts, unsigned DestReg,
                  const MCSymbol *Target, MCContext &Ctx) {
    createAuipcInstPair(Insts, DestReg, Target, RISCV::LD, Ctx);
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

  void createLI(InstructionListType &Insts, unsigned DestReg, int64_t Imm) {
    RISCVMatInt::InstSeq Seq =
        RISCVMatInt::generateInstSeq(Imm, STI->getFeatureBits());
    MCRegister SrcReg = RISCV::X0;

    for (const RISCVMatInt::Inst &Inst : Seq) {
      switch (Inst.getOpndKind()) {
      case RISCVMatInt::Imm:
        Insts.push_back(MCInstBuilder(Inst.getOpcode())
                            .addReg(DestReg)
                            .addImm(Inst.getImm()));
        break;
      case RISCVMatInt::RegX0:
        Insts.push_back(MCInstBuilder(Inst.getOpcode())
                            .addReg(DestReg)
                            .addReg(SrcReg)
                            .addReg(RISCV::X0));
        break;
      case RISCVMatInt::RegReg:
        Insts.push_back(MCInstBuilder(Inst.getOpcode())
                            .addReg(DestReg)
                            .addReg(SrcReg)
                            .addReg(SrcReg));
        break;
      case RISCVMatInt::RegImm:
        Insts.push_back(MCInstBuilder(Inst.getOpcode())
                            .addReg(DestReg)
                            .addReg(SrcReg)
                            .addImm(Inst.getImm()));
        break;
      }

      // Only the first instruction has X0 as its source.
      SrcReg = DestReg;
    }
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
      unsigned TargetReg = Jump.getOperand(0).getReg();
      createMV(Jump, DestReg, TargetReg);
    }
    }
  }

  void createRegInc(MCInst &Inst, unsigned Reg, int64_t Imm) const {
    Inst = MCInstBuilder(RISCV::ADDI).addReg(Reg).addReg(Reg).addImm(Imm);
  }

  void createSPInc(MCInst &Inst, int64_t Imm) const {
    createRegInc(Inst, RISCV::X2, Imm);
  }

  void createStore(MCInst &Inst, unsigned Reg, unsigned BaseReg,
                   int64_t Offset) const {
    Inst = MCInstBuilder(RISCV::SD).addReg(Reg).addReg(BaseReg).addImm(Offset);
  }

  void createLoad(MCInst &Inst, unsigned Reg, unsigned BaseReg,
                  int64_t Offset) const {
    Inst = MCInstBuilder(RISCV::LD).addReg(Reg).addReg(BaseReg).addImm(Offset);
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
                  const SmallVector<unsigned> &Regs) const {
    int64_t Offset = 0;
    for (auto Reg : Regs) {
      createLoad(Insts.emplace_back(), Reg, RISCV::X2, Offset);
      Offset += 8;
    }

    createSPInc(Insts.emplace_back(), Regs.size() * 8);
  }
};

} // end anonymous namespace

namespace llvm {
namespace bolt {

MCPlusBuilder *createRISCVMCPlusBuilder(const MCInstrAnalysis *Analysis,
                                        const MCInstrInfo *Info,
                                        const MCRegisterInfo *RegInfo,
                                        const MCSubtargetInfo *STI) {
  return new RISCVMCPlusBuilder(Analysis, Info, RegInfo, STI);
}

} // namespace bolt
} // namespace llvm
