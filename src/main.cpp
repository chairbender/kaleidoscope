#include <iostream>

using std::string, std::cin;

enum class Token : uint8_t {
    EndOfFile,
    Def,
    Extern,
    Identifier,
    Number
};

// TODO: do these really need to be globals?
// filled in if Identifier
static string TokenIdentifierStr;
// filled in if NumVal
static double TokenNumVal;

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

    // TODO: LEFT OFF

}

int main() {
    std::cout << "Hello, World!" << std::endl;
    return 0;
}
