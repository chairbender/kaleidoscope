// TODO: try import
#include <cassert>
#include <iostream>
#include <utility>
#include <vector>
#include <memory>
#include <string>
#include <cctype>
#include <format>
#include <unordered_map>

using std::string, std::cin, std::move, std::unique_ptr, std::vector, std::unordered_map, std::cerr,
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
  UnknownChar
};

// filled in if Identifier
static string TokenIdentifierStr;
// filled in if NumVal
static double TokenNumVal;
// filled in if UnknownChar
static char TokenUnknownChar;

// TODO: are these the best functions to use in modern C++?
static Token gettok() {
  static char LastChar{' '};

  // Skip any whitespace.
  while (isspace(LastChar))
    LastChar = cin.get();

  // identifier: a-zA-Z0-9
  if (isalpha(LastChar)) {
    TokenIdentifierStr = LastChar;
    while (isalnum(LastChar = static_cast<char>(cin.get())))
      TokenIdentifierStr += LastChar;

    if (TokenIdentifierStr == "def")
      return Token::Def;
    if (TokenIdentifierStr == "extern")
      return Token::Extern;
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
      LastChar = cin.get();
    while (LastChar != EOF && LastChar != '\n' && LastChar != '\r');

    if (LastChar != EOF)
      return Token::EndOfFile;
  }


  // check for EOF but don't consume it
  if (LastChar == EOF)
    return Token::EndOfFile;

  TokenUnknownChar = LastChar;
  LastChar = cin.get();
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
  };

  // numeric literals
  class NumberExprAst : public ExprAST {
    const double Val;

  public:
    NumberExprAst(const double Val) : Val{Val} {
    }
  };

  // variables
  class VariableExprAST : public ExprAST {
    const string Name;

  public:
    VariableExprAST(string Name) : Name{std::move(Name)} {
    }
  };

  // binary operators
  class BinaryExprAst : public ExprAST {
    const char Op;
    const unique_ptr<ExprAST> LHS, RHS;

  public:
    BinaryExprAst(const char Op, unique_ptr<ExprAST> LHS, unique_ptr<ExprAST> RHS)
      : Op{Op}, LHS{move(LHS)}, RHS{move(RHS)} {
    }
  };

  // function calls
  class CallExprAST : public ExprAST {
    const string Callee;
    const vector<unique_ptr<ExprAST> > Args;

  public:
    CallExprAST(string Callee, vector<unique_ptr<ExprAST> > Args)
      : Callee{std::move(Callee)}, Args{move(Args)} {
    }
  };

  // function prototype
  class PrototypeAST {
    const string Name;
    const vector<string> args;

  public:
    PrototypeAST(string Name, vector<string> args)
      : Name{std::move(Name)}, args{move(args)} {
    }

    [[nodiscard]] const string &getName() const { return Name; }
  };

  // function definition
  class FunctionAST {
    const unique_ptr<PrototypeAST> Proto;
    const unique_ptr<ExprAST> Body;

  public:
    FunctionAST(unique_ptr<PrototypeAST> Proto, unique_ptr<ExprAST> Body)
      : Proto{move(Proto)}, Body{move(Body)} {
    }
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

static unique_ptr<ExprAST> ParseExpression();

/// numberexpr ::= number
static unique_ptr<ExprAST> ParseNumberExpr() {
  auto Result{make_unique<NumberExprAst>(TokenNumVal)};
  getNextToken();
  return move(Result);
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
        Args.push_back(move(Arg));
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

  return make_unique<CallExprAST>(IdName, move(Args));
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
      RHS = ParseBinOpRHS(TokPrec + 1, move(RHS));
      if (!RHS)
        return nullptr;
    }

    // merge
    LHS = make_unique<BinaryExprAst>(BinOp, move(LHS), move(RHS));
  }
}

/// expression
///   ::= primary binoprhs
static unique_ptr<ExprAST> ParseExpression() {
  auto LHS{ParsePrimary()};
  if (!LHS)
    return nullptr;

  return ParseBinOpRHS(0, move(LHS));
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

  return make_unique<PrototypeAST>(Name, move(ArgsNames));
}

/// definition ::= 'def' prototype expression
static unique_ptr<FunctionAST> ParseDefinition() {
  getNextToken(); // consume def
  auto Proto{ParsePrototype()};
  if (!Proto)
    return nullptr;

  if (auto E{ParseExpression()})
    return make_unique<FunctionAST>(move(Proto), move(E));
  return nullptr;
}

/// toplevelexpr ::= expression
static unique_ptr<FunctionAST> ParseTopLevelExpr() {
  if (auto E{ParseExpression()}) {
    // anonymous proto
    auto Proto{make_unique<PrototypeAST>("__anon_expr", vector<string>{})};
    return make_unique<FunctionAST>(move(Proto), move(E));
  }
  return nullptr;
}

/// external ::= 'extern' prototype
static unique_ptr<PrototypeAST> ParseExtern() {
  getNextToken(); // consume extern
  return ParsePrototype();
}

//===-------------
// Top-level parsing
//===-------------
static void HandleDefinition() {
  if (ParseDefinition()) {
    cerr << "Parsed a function definition.\n";
  } else {
    // skip token for err recovery
    getNextToken();
  }
}

static void HandleExtern() {
  if (auto Proto = ParseExtern()) {
    cerr << "Parsed an extern declaration.\n";
  } else {
    // skip for err recovery
    getNextToken();
  }
}

static void HandleTopLevelExpression() {
  // evaluate a top-level expr into an anon function
  if (ParseTopLevelExpr()) {
    cerr << "Parsed a top-level expression.\n";
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

int main() {
  // Prime the first token.
  cerr << "ready> ";
  getNextToken();

  // Run the main "interpreter loop" now.
  MainLoop();

  return 0;
}
