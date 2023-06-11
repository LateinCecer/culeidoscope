pub mod ast;

use std::error::Error;
use std::fmt::{Display, Formatter};
use std::str::{Chars, FromStr};
use crate::lang::LexError::UnknownTokenType;

#[derive(Copy, Clone, Debug, PartialOrd, PartialEq)]
pub enum Punct {
    /// `+`
    Plus,
    /// `-`
    Minus,
    /// `*`
    Astrix,
    /// `/`
    Slash,
    /// `>`
    Gtr,
    /// `<`
    Lsr,
    /// `==`
    Eq,
    /// `>=`
    Geq,
    /// `<=`
    Leq,
    /// `=`
    Assign,
    /// `&`
    And,
    /// `|`
    Or,
    /// `^`
    Xor,
    /// `(`
    BracketOpen,
    /// `)`
    BracketClose,
    /// `;`
    Semicolon,
    /// `,`
    Comma,
    /// `!`
    Not,
    /// `!=`
    Neq,
    /// `.`
    Dot,
    IndexOpen,
    IndexClose,
}

#[derive(Clone, Debug, PartialOrd, PartialEq)]
pub enum Token {
    Eof,
    Def,
    Extern,
    Global,
    Const,
    If,
    Else,
    Then,
    For,
    In,
    Ident(String),
    Number(f64),
    Punct(Punct),
}

#[derive(Copy, Clone, Default, Debug)]
pub struct SrcPos {
    line: usize,
    col: usize,
}

impl Display for SrcPos {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "line {}, col {}", self.line, self.col)
    }
}

#[derive(Debug)]
pub enum LexError {
    UnexpectedEndOfFile(SrcPos),
    UnknownTokenType(SrcPos, String),
    MalformedLiteral(SrcPos, String),
    WrongTokenType(SrcPos, Token, Token),
}

impl Display for LexError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            LexError::UnexpectedEndOfFile(pos) => write!(f, "Unexpected end of file @ {pos}"),
            LexError::UnknownTokenType(pos, tok) => write!(f, "Unexpected token '{tok:?}' @ {pos}"),
            LexError::MalformedLiteral(pos, lit) => write!(f, "Malformed literal '{lit}' @ {pos}"),
            LexError::WrongTokenType(pos, got, exp) => write!(f, "Wrong token type @ {pos}. Got token {got:?} where {exp:?} was expected")
        }
    }
}

impl Error for LexError {}

pub struct Lexer<'a> {
    last_char: Option<char>,
    src: Chars<'a>,
    pos: SrcPos,
}

impl<'a> Lexer<'a> {
    pub fn new(src: Chars<'a>) -> Self {
        Lexer {
            last_char: Some(' '),
            src,
            pos: SrcPos::default(),
        }
    }

    pub fn get_tok(&mut self) -> Result<Token, LexError> {
        // skip whitespace
        self.skip_whitespace();
        // check for identifiers
        if self.check_last(char::is_alphabetic) {
            let mut lit = String::new();
            lit.push(self.last_or()?);

            self.pop();
            while self.check_last(char::is_alphanumeric) {
                lit.push(self.last_or()?);
                self.pop();
            }

            return if lit == "def" {
                Ok(Token::Def)
            } else if lit == "extern" {
                Ok(Token::Extern)
            } else if lit == "global" {
                Ok(Token::Global)
            } else if lit == "const" {
                Ok(Token::Const)
            } else if lit == "if" {
                Ok(Token::If)
            } else if lit == "then" {
                Ok(Token::Then)
            } else if lit == "else" {
                Ok(Token::Else)
            } else if lit == "for" {
                Ok(Token::For)
            } else if lit == "in" {
                Ok(Token::In)
            } else {
                Ok(Token::Ident(lit))
            };
        }

        // check for numbers
        if self.check_last(|c| c.is_ascii_digit()) {
            let mut num_str = String::new();
            num_str.push(self.last_or()?);
            self.pop();

            while self.check_last(|c| c.is_ascii_digit() || c == '.') {
                num_str.push(self.last_or()?);
                self.pop();
            }

            return Ok(Token::Number(f64::from_str(&num_str)
                .map_err(|_| LexError::MalformedLiteral(self.pos, num_str))?));
        }

        // check for line comments
        if self.check_last(|c| c == '#') {
            // comment until end of line
            self.pop();
            while self.check_last(|c| c != '\n' && c != '\r') {
                self.pop();
            }
            return self.get_tok();
        }

        match self.last_char {
            Some('+') => {
                self.pop();
                Ok(Token::Punct(Punct::Plus))
            },
            Some('-') => {
                self.pop();
                Ok(Token::Punct(Punct::Minus))
            },
            Some('*') => {
                self.pop();
                Ok(Token::Punct(Punct::Astrix))
            },
            Some('/') => {
                self.pop();
                Ok(Token::Punct(Punct::Slash))
            },
            Some('|') => {
                self.pop();
                Ok(Token::Punct(Punct::Or))
            },
            Some('&') => {
                self.pop();
                Ok(Token::Punct(Punct::And))
            },
            Some('^') => {
                self.pop();
                Ok(Token::Punct(Punct::Xor))
            },
            Some('<') => {
                self.pop();
                if self.check_last(|c| c == '=') {
                    Ok(Token::Punct(Punct::Leq))
                } else {
                    Ok(Token::Punct(Punct::Lsr))
                }
            },
            Some('>') => {
                self.pop();
                if self.check_last(|c| c == '=') {
                    Ok(Token::Punct(Punct::Geq))
                } else {
                    Ok(Token::Punct(Punct::Gtr))
                }
            },
            Some(';') => {
                self.pop();
                Ok(Token::Punct(Punct::Semicolon))
            },
            Some(',') => {
                self.pop();
                Ok(Token::Punct(Punct::Comma))
            },
            Some('=') => {
                self.pop();
                if self.check_last(|c| c == '=') {
                    Ok(Token::Punct(Punct::Eq))
                } else {
                    Ok(Token::Punct(Punct::Assign))
                }
            },
            Some('!') => {
                self.pop();
                if self.check_last(|c| c == '=') {
                    Ok(Token::Punct(Punct::Neq))
                } else {
                    Ok(Token::Punct(Punct::Not))
                }
            },
            Some('(') => {
                self.pop();
                Ok(Token::Punct(Punct::BracketOpen))
            },
            Some(')') => {
                self.pop();
                Ok(Token::Punct(Punct::BracketClose))
            },
            Some('.') => {
                self.pop();
                Ok(Token::Punct(Punct::Dot))
            },
            Some('[') => {
                self.pop();
                Ok(Token::Punct(Punct::IndexOpen))
            },
            Some(']') => {
                self.pop();
                Ok(Token::Punct(Punct::IndexClose))
            },
            Some(c) => Err(UnknownTokenType(self.pos, format!("{}", c))),
            None => Ok(Token::Eof)
        }
    }

    fn check_last<F: Fn(char) -> bool>(&self, f: F) -> bool {
        if let Some(c) = self.last_char.clone() {
            f(c)
        } else {
            false
        }
    }

    #[inline(always)]
    fn last_or(&self) -> Result<char, LexError> {
        self.last_char.ok_or(LexError::UnexpectedEndOfFile(self.pos))
    }

    fn skip_whitespace(&mut self) {
        while self.check_last(char::is_whitespace) {
            self.pop();
        }
    }

    fn pop(&mut self) {
        self.last_char = self.src.next();
        self.pos.col += 1;
        if self.last_char == Some('\n') {
            self.pos.line += 1;
            self.pos.col = 0;
        }
    }
}
