// TODO: try import
#include "KaleidoscopeJIT.hpp"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/StandardInstrumentations.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Scalar/GVN.h"
#include "llvm/Transforms/Scalar/Reassociate.h"
#include "llvm/Transforms/Scalar/SimplifyCFG.h"
#include <cassert>
#include <iostream>
#include <utility>
#include <vector>
#include <memory>
#include <string>
#include <cctype>
#include <format>
#include <map>
#include <unordered_map>

using namespace llvm;

using std::string, std::cin, std::unique_ptr, std::vector, std::unordered_map, std::cerr,
    std::format, std::make_unique;

//===-------------
// Lexer
//===-------------

enum class Token : uint8_t {
  EndOfFile,
  Def,
  Extern,
  Identifier,
  Number,
  // note instead of this, the tutorial returns an int from gettok and uses a convention
  // of negative values = known tokens and positive = unknown char.
  // For readability / expressing intent better, I'm trying out this approach instead of actually
  // handling an unknown char as yet another token type.
  UnknownChar,
  If,
  Then,
  Else,
  For,
  In
};

// filled in if Identifier
static string TokenIdentifierStr;
// filled in if NumVal
static double TokenNumVal;
// filled in if UnknownChar
static char TokenUnknownChar;

static char getaschar() {
  return static_cast<char>(cin.get());
}

static Token gettok() {
  static char LastChar{' '};

  // Skip any whitespace.
  while (isspace(LastChar))
    LastChar = getaschar();

  // identifier: a-zA-Z0-9
  if (isalpha(LastChar)) {
    TokenIdentifierStr = LastChar;
    while (isalnum(LastChar = getaschar()))
      TokenIdentifierStr += LastChar;

    if (TokenIdentifierStr == "def")
      return Token::Def;
    if (TokenIdentifierStr == "extern")
      return Token::Extern;
    if (TokenIdentifierStr == "if")
      return Token::If;
    if (TokenIdentifierStr == "then")
      return Token::Then;
    if (TokenIdentifierStr == "else")
      return Token::Else;
    if (TokenIdentifierStr == "for")
      return Token::For;
    if (TokenIdentifierStr == "in")
      return Token::In;
    return Token::Identifier;
  }

  // Number: 0-9
  if (isdigit(LastChar) || LastChar == '.') {
    string NumStr;
    do {
      NumStr += LastChar;
      cin >> LastChar;
    } while (isdigit(LastChar) || LastChar == '.');
    TokenNumVal = stod(NumStr);
    return Token::Number;
  }

  // comment until end of line
  if (LastChar == '#') {
    do
      LastChar = getaschar();
    while (LastChar != EOF && LastChar != '\n' && LastChar != '\r');

    if (LastChar != EOF)
      return Token::EndOfFile;
  }


  // check for EOF but don't consume it
  if (LastChar == EOF)
    return Token::EndOfFile;

  TokenUnknownChar = LastChar;
  LastChar = getaschar();
  return Token::UnknownChar;
}

//===-------------
// AST (aka Parse Tree)
//===-------------
namespace {
  // base class for all expression nodes
  class ExprAST {
  public:
    virtual ~ExprAST() = default;
    // note this is a raw pointer because it refers to something actually owned by TheModule
    // TODO: this is what the tutorial says, but why not use a shared ptr instead?
    virtual Value* codegen() = 0;
  };

  // numeric literals
  class NumberExprAst : public ExprAST {
    const double Val;

  public:
    NumberExprAst(const double Val) : Val{Val} {}
    Value* codegen() override;
  };

  // variables
  class VariableExprAST : public ExprAST {
    const string Name;

  public:
    VariableExprAST(string Name) : Name{std::move(Name)} {}
    Value* codegen() override;
  };

  // binary operators
  class BinaryExprAst : public ExprAST {
    const char Op;
    const unique_ptr<ExprAST> LHS, RHS;

  public:
    BinaryExprAst(const char Op, unique_ptr<ExprAST> LHS, unique_ptr<ExprAST> RHS)
      : Op{Op}, LHS{std::move(LHS)}, RHS{std::move(RHS)} {}
    Value* codegen() override;
  };

  // function calls
  class CallExprAST : public ExprAST {
    const string Callee;
    const vector<unique_ptr<ExprAST> > Args;

  public:
    CallExprAST(string Callee, vector<unique_ptr<ExprAST> > Args)
      : Callee{std::move(Callee)}, Args{std::move(Args)} {}
    Value* codegen() override;
  };

  // if/then/else
  class IfExprAST : public ExprAST {
    const unique_ptr<ExprAST> Cond, Then, Else;
  public:
    IfExprAST(unique_ptr<ExprAST> Cond, unique_ptr<ExprAST> Then, unique_ptr<ExprAST> Else)
      : Cond{std::move(Cond)}, Then{std::move(Then)}, Else{std::move(Else)} {}
    Value* codegen() override;
  };

  // for
  class ForExprAST : public ExprAST {
    const string VarName;
    const unique_ptr<ExprAST> Start, End, Step;
    const unique_ptr<ExprAST> Body;
  public:
    ForExprAST(string VarName, unique_ptr<ExprAST> Start, unique_ptr<ExprAST> End, unique_ptr<ExprAST> Step, unique_ptr<ExprAST> Body)
      : VarName{std::move(VarName)}, Start{std::move(Start)}, End{std::move(End)},
    Step{std::move(Step)}, Body{std::move(Body)} {}

    Value* codegen() override;
  };

  // function prototype
  class PrototypeAST {
    const string Name;
    const vector<string> Args;

  public:
    PrototypeAST(string Name, vector<string> Args)
      : Name{std::move(Name)}, Args{std::move(Args)} {}
    [[nodiscard]] Function *codegen() const;
    [[nodiscard]] const string &getName() const { return Name; }
  };

  // function definition
  class FunctionAST {
    unique_ptr<PrototypeAST> Proto;
    const unique_ptr<ExprAST> Body;

  public:
    FunctionAST(unique_ptr<PrototypeAST> Proto, unique_ptr<ExprAST> Body)
      : Proto{std::move(Proto)}, Body{std::move(Body)} {}
    Function *codegen();
  };
}

//===-------------
// Parser
//===-------------
// current token the parser is looking at
static Token CurTok;
// read another token from lexer and update CurTok
static Token getNextToken() { return CurTok = gettok(); }

static bool CurUnknownCharIs(const char C) {
  return CurTok == Token::UnknownChar && TokenUnknownChar == C;
}

// precedence for each binary operator
static const std::unordered_map<char, int> BinopPrecedence = {
  {'<', 10},
  {'+', 20},
  {'-', 20},
  {'*', 40}
};

// get the precedence of the current binary operator token
static int GetTokPrecedence() {
  if (CurTok != Token::UnknownChar)
    return -1;

  const auto TokPrec{BinopPrecedence.find(TokenUnknownChar)};
  if (TokPrec == BinopPrecedence.end())
    return -1;
  return TokPrec->second;
}

template<typename T>
unique_ptr<T> LogError(const string &Str) {
  cerr << std::format("Error: {}\n", Str);
  return nullptr;
}

Value* LogErrorV(const string &Str) {
  cerr << std::format("Error: {}\n", Str);
  return nullptr;
}

static unique_ptr<ExprAST> ParseExpression();

/// numberexpr ::= number
static unique_ptr<ExprAST> ParseNumberExpr() {
  auto Result{make_unique<NumberExprAst>(TokenNumVal)};
  getNextToken();
  return std::move(Result);
}

/// parenexpr ::= '(' expression ')'
static unique_ptr<ExprAST> ParseParenExpr() {
  getNextToken(); // consume (
  auto V{ParseExpression()};
  if (!V)
    return nullptr;

  if (!CurUnknownCharIs(')'))
    return LogError<ExprAST>("expected ')'");
  getNextToken(); // consume )
  return V;
}

/// identifierexpr
///   ::= identifier
///   ::= identifier '(' expression* ')'
static unique_ptr<ExprAST> ParseIdentifierExpr() {
  const auto IdName{TokenIdentifierStr};

  getNextToken(); // consume identifier

  // variable ref
  if (!CurUnknownCharIs('('))
    return make_unique<VariableExprAST>(IdName);

  // call
  getNextToken(); // consume (
  vector<unique_ptr<ExprAST>> Args;
  if (!CurUnknownCharIs(')')) {
    while (true) {
      if (auto Arg{ParseExpression()})
        Args.push_back(std::move(Arg));
      else
        return nullptr;

      if (CurUnknownCharIs(')'))
        break;

      if (!CurUnknownCharIs(','))
        return LogError<ExprAST>("expected ')' or ',' in argument list");
      getNextToken();
    }
  }

  getNextToken(); // consume )

  return make_unique<CallExprAST>(IdName, std::move(Args));
}

/// ifexpr ::= 'if' expression 'then' expression 'else' expression
static unique_ptr<ExprAST> ParseIfExpr() {
  getNextToken(); // consume if

  // condition
  auto Cond = ParseExpression();
  if (!Cond)
    return nullptr;

  if (CurTok != Token::Then)
    return LogError<ExprAST>("expected then");
  getNextToken(); // consume then

  auto Then = ParseExpression();
  if (!Then)
    return nullptr;

  if (CurTok != Token::Else)
    return LogError<ExprAST>("expected else");
  getNextToken(); // consume else

  auto Else = ParseExpression();
  if (!Else)
    return nullptr;

  return make_unique<IfExprAST>(std::move(Cond), std::move(Then), std::move(Else));
}

/// forexpr ::= 'for' identifire '=' expr ',' expr (',' expr)? 'in' expression
static unique_ptr<ExprAST> ParseForExpr() {
  getNextToken(); // consume for

  if (CurTok != Token::Identifier)
    return LogError<ExprAST>("expected identifier in for loop");

  auto VarName{TokenIdentifierStr};
  getNextToken(); // consume identifier

  if (!CurUnknownCharIs('='))
    return LogError<ExprAST>("expected '=' after for loop variable");
  getNextToken(); // consume =

  auto Start{ParseExpression()};
  if (!Start)
    return nullptr;
  if (!CurUnknownCharIs(','))
    return LogError<ExprAST>("expected ',' after for loop start");
  getNextToken(); // consume ,

  auto End{ParseExpression()};
  if (!End)
    return nullptr;

  // optional step value
  unique_ptr<ExprAST> Step;
  if (CurUnknownCharIs(',')) {
    getNextToken(); // consume ,
    Step = ParseExpression();
    if (!Step)
      return nullptr;
  }

  if (CurTok != Token::In)
    return LogError<ExprAST>("expected 'in' after for loop");
  getNextToken(); // consume in

  auto Body{ParseExpression()};
  if (!Body)
    return nullptr;

  return make_unique<ForExprAST>(VarName, std::move(Start), std::move(End), std::move(Step), std::move(Body));
}

/// primary
///   ::= identifierexpr
///   ::= numberexpr
///   ::= parenexpr
static unique_ptr<ExprAST> ParsePrimary() {
  switch (CurTok) {
    default:
      if (CurUnknownCharIs('('))
        return ParseParenExpr();
      return LogError<ExprAST>("unknown token when expecting an expression");
    case Token::Identifier:
      return ParseIdentifierExpr();
    case Token::Number:
      return ParseNumberExpr();
    case Token::If:
      return ParseIfExpr();
    case Token::For:
      return ParseForExpr();
  }
}

/// binoprhs
///   ::= ('+' primary)*
static unique_ptr<ExprAST> ParseBinOpRHS(int ExprPrec, unique_ptr<ExprAST> LHS) {

  // if this is a binop, find its precedence
  while (true) {
    const auto TokPrec{GetTokPrecedence()};

    // if it binds at least as tightly as the current binop,
    // consume it, otherwise we are done
    if (TokPrec < ExprPrec)
      return LHS;

    assert(CurTok == Token::UnknownChar);
    const auto BinOp{TokenUnknownChar};
    getNextToken();

    auto RHS{ParsePrimary()};
    if (!RHS)
      return nullptr;

    // if binop binds less tightly with RHS than the operator after RHS,
    // let the pending operater take RHS as its LHS
    const int NextPrec{GetTokPrecedence()};
    if (TokPrec < NextPrec) {
      RHS = ParseBinOpRHS(TokPrec + 1, std::move(RHS));
      if (!RHS)
        return nullptr;
    }

    // merge
    LHS = make_unique<BinaryExprAst>(BinOp, std::move(LHS), std::move(RHS));
  }
}

/// expression
///   ::= primary binoprhs
static unique_ptr<ExprAST> ParseExpression() {
  auto LHS{ParsePrimary()};
  if (!LHS)
    return nullptr;

  return ParseBinOpRHS(0, std::move(LHS));
}

/// prototype
///   ::= id '(' id* ')'
static unique_ptr<PrototypeAST> ParsePrototype() {
  if (CurTok != Token::Identifier)
    return LogError<PrototypeAST>("expected function name in prototype");
  const auto Name{TokenIdentifierStr};
  getNextToken(); // consume name

  if (!CurUnknownCharIs('('))
    return LogError<PrototypeAST>("expected '(' in prototype");

  vector<string> ArgsNames;
  while (getNextToken() == Token::Identifier)
    ArgsNames.push_back(TokenIdentifierStr);
  if (!CurUnknownCharIs(')'))
    return LogError<PrototypeAST>("expected ')' in prototype");

  getNextToken(); // consume ')'

  return make_unique<PrototypeAST>(Name, std::move(ArgsNames));
}

/// definition ::= 'def' prototype expression
static unique_ptr<FunctionAST> ParseDefinition() {
  getNextToken(); // consume def
  auto Proto{ParsePrototype()};
  if (!Proto)
    return nullptr;

  if (auto E{ParseExpression()})
    return make_unique<FunctionAST>(std::move(Proto), std::move(E));
  return nullptr;
}

/// toplevelexpr ::= expression
static unique_ptr<FunctionAST> ParseTopLevelExpr() {
  if (auto E{ParseExpression()}) {
    // anonymous proto
    auto Proto{make_unique<PrototypeAST>("__anon_expr", vector<string>{})};
    return make_unique<FunctionAST>(std::move(Proto), std::move(E));
  }
  return nullptr;
}

/// external ::= 'extern' prototype
static unique_ptr<PrototypeAST> ParseExtern() {
  getNextToken(); // consume extern
  return ParsePrototype();
}

//===-------------
// Code Generation
//===-------------

static unique_ptr<LLVMContext> TheContext;
static unique_ptr<IRBuilder<>> Builder;
static unique_ptr<Module> TheModule;
// holds named values which are in current scope.
// this is NOT a map of ALL the named values in the program.
static std::map<string, Value*> NamedValues;
static unique_ptr<orc::KaleidoscopeJIT> TheJIT;
static unique_ptr<FunctionPassManager> TheFPM;
static unique_ptr<LoopAnalysisManager> TheLAM;
static unique_ptr<FunctionAnalysisManager> TheFAM;
static unique_ptr<CGSCCAnalysisManager> TheCGAM;
static unique_ptr<ModuleAnalysisManager> TheMAM;
static unique_ptr<PassInstrumentationCallbacks> ThePIC;
static unique_ptr<StandardInstrumentations> TheSI;
static std::map<string, unique_ptr<PrototypeAST>> FunctionProtos;
static ExitOnError ExitOnErr;

Function* getFunction(const string &Name) {
  // see if function has already been added to the module
  if (auto* F{TheModule->getFunction(Name)})
    return F;

  // If not, check whether we can codegen the declaration
  // from the prototype
  auto FI{FunctionProtos.find(Name)};
  if (FI != FunctionProtos.end())
    return FI->second->codegen();

  return nullptr;
}


// todo: visitor pattern might be better instead of codegen-ing
//  directly from the AST, but this is simpler for a tutorial

Value* NumberExprAst::codegen() {
  return ConstantFP::get(*TheContext, APFloat(Val));
}

Value* VariableExprAST::codegen() {
  const auto V{NamedValues[Name]};
  if (!V)
    LogErrorV("Unknown variable name");
  return V;
}

Value* BinaryExprAst::codegen() {
  const auto L{LHS->codegen()};
  const auto R{RHS->codegen()};
  if (!L || !R)
    return nullptr;

  switch (Op) {
    case '+':
      return Builder->CreateFAdd(L, R, "addtmp");
    case '-':
      return Builder->CreateFSub(L, R, "subtmp");
    case '*':
      return Builder->CreateFMul(L, R, "multmp");
    case '<': {
      const auto cmp = Builder->CreateFCmpULT(L, R, "cmptmp");
      // convert bool 0/1 to double 0.0 or 1.0
      return Builder->CreateUIToFP(cmp, Type::getDoubleTy(*TheContext), "booltmp");
    }
    default:
      return LogErrorV("invalid binary operator");
  }
}

Value* CallExprAST::codegen() {
  const auto CalleeF{getFunction(Callee)};
  if (!CalleeF)
    return LogErrorV("Unknown function referenced");

  // if argument mismatch error
  if (CalleeF->arg_size() != Args.size())
    return LogErrorV("Incorrect # arguments passed");

  vector<Value*> ArgsV;
  for (const auto& arg : Args) {
    ArgsV.push_back(arg->codegen());
    if (!ArgsV.back())
      return nullptr;
  }

  return Builder->CreateCall(CalleeF, ArgsV, "calltmp");
}

Value* IfExprAST::codegen() {
  auto CondV{Cond->codegen()};
  if (!CondV)
    return nullptr;

  // convert condition to a bool by comparing neq to 0.0
  CondV = Builder->CreateFCmpONE(
    CondV, ConstantFP::get(*TheContext, APFloat(0.0)), "ifcond");

  const auto TheFunction{Builder->GetInsertBlock()->getParent()};

  // create blocks for the then and else cases. Insert then block at the end of the function
  auto ThenBB{BasicBlock::Create(*TheContext, "then", TheFunction)};
  auto ElseBB{BasicBlock::Create(*TheContext, "else")};
  const auto MergeBB{BasicBlock::Create(*TheContext, "ifcont")};

  Builder->CreateCondBr(CondV, ThenBB, ElseBB);

  // emit then block
  Builder->SetInsertPoint(ThenBB);
  const auto ThenV{Then->codegen()};
  if (!ThenV)
    return nullptr;

  Builder->CreateBr(MergeBB);
  // because codegen of "then" can change the current block (for example for nested if/then), update ThenBB for the PHI
  // so it points at the possibly new block
  ThenBB = Builder->GetInsertBlock();

  // emit else block
  TheFunction->insert(TheFunction->end(), ElseBB);
  Builder->SetInsertPoint(ElseBB);

  const auto ElseV{Else->codegen()};
  if (!ElseV)
    return nullptr;

  Builder->CreateBr(MergeBB);
  // because codegen of "else" can change the current block (for example for nested if/then), update ElseBB for the PHI
  // so it points at the possibly new block
  ElseBB = Builder->GetInsertBlock();

  // emit merge block
  TheFunction->insert(TheFunction->end(), MergeBB);
  Builder->SetInsertPoint(MergeBB);
  auto PN{Builder->CreatePHI(Type::getDoubleTy(*TheContext), 2, "iftmp")};

  PN->addIncoming(ThenV, ThenBB);
  PN->addIncoming(ElseV, ElseBB);
  return PN;
}

Value *ForExprAST::codegen() {
  // emit start code first, without variable in scope
  const auto StartValue{Start->codegen()};
  if (!StartValue)
    return nullptr;

  // Make new basic block for the loop header, inserting after current block
  const auto TheFunction{Builder->GetInsertBlock()->getParent()};
  const auto PreheaderBB{Builder->GetInsertBlock()};
  const auto LoopBB{BasicBlock::Create(*TheContext, "loop", TheFunction)};

  // insert explicit fallthrough from current block to the LoopBB (for the upcoming phi)
  Builder->CreateBr(LoopBB);

  // Start insertion in LoopBB
  Builder->SetInsertPoint(LoopBB);

  // Start the PHI node with an entry for Start.
  // We haven't generated the loop body yet so we'll have to wait to add that to the phi
  const auto Variable{Builder->CreatePHI(Type::getDoubleTy(*TheContext), 2, VarName)};
  Variable->addIncoming(StartValue, PreheaderBB);

  // within the loop, the variable is defined equal to the phi node.
  // If it shadows an existing var, we hae to restore it, so save it now.
  const auto OldVal{NamedValues[VarName]};
  NamedValues[VarName] = Variable;

  // Emit the body of the loop. This, like any other expr, can change the
  // current BB.
  // Note that we ignore the value computed by the body, but don't allow an error.
  if (!Body->codegen())
    return nullptr;

  // emit the step value
  Value* StepValue = nullptr;
  if (Step) {
    StepValue = Step->codegen();
    if (!StepValue)
      return nullptr;
  } else {
    // if no step value, use 1.0
    StepValue = ConstantFP::get(*TheContext, APFloat(1.0));
  }

  const auto NextVar{Builder->CreateFAdd(Variable, StepValue, "nextvar")};

  // compute end condition
  auto EndCond{End->codegen()};
  if (!EndCond)
    return nullptr;

  // convert condition to a bool by comparing neq 0.0
  EndCond = Builder->CreateFCmpONE(
    EndCond, ConstantFP::get(*TheContext, APFloat(0.0)), "loopcond");

  // create the "after loop" block and insert it
  const auto LoopEndBB{Builder->GetInsertBlock()};
  const auto AfterBB{BasicBlock::Create(*TheContext, "afterloop", TheFunction)};

  // insert conditional branch into end of LoopEndBB
  Builder->CreateCondBr(EndCond, LoopBB, AfterBB);

  // any new code will be inserted in AfterBB
  Builder->SetInsertPoint(AfterBB);

  // add a new entry to the phi node for the backedge
  Variable->addIncoming(NextVar, LoopEndBB);

  //restore unshadowed var
  if (OldVal)
    NamedValues[VarName] = OldVal;
  else
    NamedValues.erase(VarName);

  // for expr always returns 0.0
  return Constant::getNullValue(Type::getDoubleTy(*TheContext));
}

Function* PrototypeAST::codegen() const {
  const vector Doubles{Args.size(), Type::getDoubleTy(*TheContext)};
  const auto FT{FunctionType::get(Type::getDoubleTy(*TheContext), Doubles, false)};
  const auto F{Function::Create(FT, Function::ExternalLinkage, Name, TheModule.get())};
  // note: not strictly necessary - we can let llvm generate the names for us,
  // but helps with readability of the IR
  unsigned Idx = 0;
  for (auto& Arg : F->args())
    Arg.setName(Args[Idx++]);

  return F;
}

Function* FunctionAST::codegen() {
  // transfer ownership of the prototype to the functionprotos map,
  // but keep a reference to it for use below
  const auto name = Proto->getName();
  FunctionProtos[Proto->getName()] = std::move(Proto);
  Function* TheFunction = getFunction(name);
  if (!TheFunction)
    return nullptr;

  //create a new BB to start insertion into
  const auto BB{BasicBlock::Create(*TheContext, "entry", TheFunction)};
  Builder->SetInsertPoint(BB);

  // record function args in the NamedValues map
  // so we can reference them later in this same scope
  NamedValues.clear();
  for (auto& Arg : TheFunction->args())
    NamedValues[string{Arg.getName()}] = &Arg;

  if (const auto RetVal{Body->codegen()}) {
      // finish off the function
      Builder->CreateRet(RetVal);

      // ask LLVM to check for consistency
      verifyFunction(*TheFunction);

      // Run the optimizer on it (in place)
      TheFPM->run(*TheFunction, *TheFAM);

      return TheFunction;
  }

  // error reading body, remove function
  TheFunction->eraseFromParent();
  return nullptr;
}

//===-------------
// Top-level parsing and JIT Driver
//===-------------
static void InitializeModuleAndManagers() {
  // open a new context and module
  TheContext = make_unique<LLVMContext>();
  TheModule = make_unique<Module>("my cool jit", *TheContext);
  TheModule->setDataLayout(TheJIT->getDataLayout());

  Builder = make_unique<IRBuilder<>>(*TheContext);

  // create new transform and analysis pass managers
  TheFPM = make_unique<FunctionPassManager>();
  TheLAM = make_unique<LoopAnalysisManager>();
  TheFAM = make_unique<FunctionAnalysisManager>();
  TheCGAM = make_unique<CGSCCAnalysisManager>();
  TheMAM = make_unique<ModuleAnalysisManager>();
  ThePIC = make_unique<PassInstrumentationCallbacks>();
  TheSI = make_unique<StandardInstrumentations>(*TheContext, true);

  TheSI->registerCallbacks(*ThePIC, TheMAM.get());

  // add transform passes
  // simple peephole / bit-twiddling optzns
  TheFPM->addPass(InstCombinePass());
  // reassociate expressions
  TheFPM->addPass(ReassociatePass());
  // eliminate common subexpressions
  TheFPM->addPass(GVNPass());
  // simplify the control flow graph (deleting unreachable blocks, etc)
  TheFPM->addPass(SimplifyCFGPass());


  // register analysis passes used in these transform passes
  PassBuilder PB;
  PB.registerModuleAnalyses(*TheMAM);
  PB.registerFunctionAnalyses(*TheFAM);
  // basically seems like it makes all the managers "aware" of each other.
  PB.crossRegisterProxies(*TheLAM, *TheFAM, *TheCGAM, *TheMAM);
}

static void HandleDefinition() {
  if (const auto FnAST = ParseDefinition()) {
    if (const auto FnIR = FnAST->codegen()) {
      cerr << "Read function definition:\n";
      FnIR->print(errs());
      cerr << "\n\n";
      ExitOnErr(TheJIT->addModule(
        orc::ThreadSafeModule(std::move(TheModule),
          std::move(TheContext))));
      InitializeModuleAndManagers();
    }
  } else {
    // skip token for err recovery
    getNextToken();
  }
}

static void HandleExtern() {
  if (auto ProtoAST = ParseExtern()) {
    if (const auto FnIR = ProtoAST->codegen()) {
      cerr << "Read extern:\n";
      FnIR->print(errs());
      cerr << "\n\n";
      FunctionProtos[ProtoAST->getName()] = std::move(ProtoAST);
    }
  } else {
    // skip for err recovery
    getNextToken();
  }
}

static void HandleTopLevelExpression() {
  // evaluate a top-level expr into an anon function
  if (const auto FnAST = ParseTopLevelExpr()) {
    if (FnAST->codegen()) {
      // Track the JIT'd memory allocated to our anonymous
      // expression so we can free it after execution
      auto RT{TheJIT->getMainJITDylib().createResourceTracker()};

      auto TSM{orc::ThreadSafeModule(std::move(TheModule), std::move(TheContext))};
      ExitOnErr(TheJIT->addModule(std::move(TSM), RT));
      InitializeModuleAndManagers();

      // search JIT for the anon expr symbol
      const auto ExprSymbol{ExitOnErr(TheJIT->lookup("__anon_expr"))};
      // TODO: doesn't compile with below - out of date tutorial?
      //assert(ExprSymbol && "Function not found");

      // get symbol's addres and cast it to the right type
      // (takes no args, returns a double) so we can call it
      // as a function
      double (*FP)() = ExprSymbol.getAddress().toPtr<double (*)()>();
      // note we can now call our jit compiled code directly
      // as if it was a normal function
      cerr << std::format("Evaluated expression to {}\n", FP());

      // delete the anon expression module from the JIT.
      ExitOnErr(RT->remove());
    }
  } else {
    // skip for err recovery
    getNextToken();
  }
}

/// top ::= definition | external | expression | ';'
static void MainLoop() {
  while (true) {
    cerr << "ready> ";
    switch (CurTok) {
      case Token::EndOfFile:
        return;
      case Token::Def:
        HandleDefinition();
        break;
      case Token::Extern:
        HandleExtern();
        break;
      default:
        // ignore top level semicolons
        if (CurUnknownCharIs(';')) {
          getNextToken();
          break;
        }
        HandleTopLevelExpression();
        break;
    }
  }
}

//===----------------------------------------------------------------------===//
// "Library" functions that can be "extern'd" from user code.
//===----------------------------------------------------------------------===//

#ifdef _WIN32
#define DLLEXPORT __declspec(dllexport)
#else
#define DLLEXPORT
#endif

/// putchard - putchar that takes a double and returns 0.
extern "C" DLLEXPORT double putchard(double X) {
  fputc((char)X, stderr);
  return 0;
}

/// printd - printf that takes a double prints it as "%f\n", returning 0.
extern "C" DLLEXPORT double printd(double X) {
  fprintf(stderr, "%f\n", X);
  return 0;
}

//===
// Main Driver code
//===
int main() {
  InitializeNativeTarget();
  InitializeNativeTargetAsmPrinter();
  InitializeNativeTargetAsmParser();

  // Prime the first token.
  cerr << "ready> ";
  getNextToken();

  TheJIT = ExitOnErr(orc::KaleidoscopeJIT::Create());

  InitializeModuleAndManagers();

  // Run the main "interpreter loop" now.
  MainLoop();

  return 0;
}
