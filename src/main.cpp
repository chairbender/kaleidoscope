// TODO: try import
#include <iostream>
#include <utility>
#include <vector>

using std::string, std::cin, std::move, std::unique_ptr, std::vector;

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

// TODO: do these really need to be globals?
// filled in if Identifier
static string TokenIdentifierStr;
// filled in if NumVal
static double TokenNumVal;
// filled in if UnknownChar
static int TokenUnknownChar;

// TODO: are these the best functions to use in modern C++?
static Token gettok() {
    static int LastChar = ' ';

    // this skips whitespace automatically
    // TODO: what happens on EOF?
    cin >> LastChar;

    // identifier: a-zA-Z0-9
    if (isalpha(LastChar)) {
        TokenIdentifierStr = static_cast<char>(LastChar);
        while (isalnum(LastChar = cin.get()))
            TokenIdentifierStr += static_cast<char>(LastChar);

        if (TokenIdentifierStr == "def")
            return Token::Identifier;
        if (TokenIdentifierStr == "extern")
            return Token::Extern;
        return Token::Identifier;
    }

    // Number: 0-9
    if (isdigit(LastChar) || LastChar == '.') {
        string NumStr;
        do {
            NumStr += static_cast<char>(LastChar);
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
        NumberExprAst(const double Val) : Val{Val} {}
    };

    // variables
    class VariableExprAST : public ExprAST {
        const string Name;
    public:
        VariableExprAST(string Name) : Name{std::move(Name)} {}
    };

    // binary operators
    class BinaryExprAst : public ExprAST {
        const char Op;
        const unique_ptr<ExprAST> LHS, RHS;
    public:
        BinaryExprAst(const char Op, unique_ptr<ExprAST> LHS, unique_ptr<ExprAST> RHS)
            : Op{Op}, LHS{move(LHS)}, RHS{move(RHS)} {}
    };

    // function calls
    class CallExprAST : public ExprAST {
        const string Callee;
        const vector<unique_ptr<ExprAST>> Args;
    public:
        CallExprAST(string Callee, vector<unique_ptr<ExprAST>> Args)
            : Callee{std::move(Callee)}, Args{move(Args)} {}
    };

    // function prototype
    class PrototypeAST {
        const string Name;
        const vector<string> args;
    public:
        PrototypeAST(string Name, vector<string> args)
            : Name{std::move(Name)}, args{move(args)} {}
        [[nodiscard]] const string& getName() const { return Name; }
    };

    // function definition
    class FunctionAST {
        const unique_ptr<PrototypeAST> Proto;
        const unique_ptr<ExprAST> Body;
    public:
        FunctionAST(unique_ptr<PrototypeAST> Proto, unique_ptr<ExprAST> Body)
                : Proto{move(Proto)}, Body{move(Body)} {}
    };
}

int main() {
    std::cout << "Hello, World!" << std::endl;
    return 0;
}
