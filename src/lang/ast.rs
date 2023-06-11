use std::collections::HashMap;
use std::path::Path;
use inkwell::builder::Builder;
use inkwell::context::Context;
use inkwell::{AddressSpace, FloatPredicate, OptimizationLevel};
use inkwell::attributes::{Attribute, AttributeLoc};
use inkwell::module::{Linkage, Module};
use inkwell::passes::PassManagerBuilder;
use inkwell::passes::PassManager;
use inkwell::targets::{CodeModel, FileType, InitializationConfig, RelocMode, Target, TargetTriple};
use inkwell::values::BasicValueEnum;
use inkwell::values::BasicMetadataValueEnum;
use inkwell::values::AnyValue;
use inkwell::values::FunctionValue;
use crate::lang::{Lexer, LexError, Punct, Token};

#[allow(dead_code)]
#[derive(Debug, PartialOrd, PartialEq)]
pub enum Intrinsic {
    TidX,
    TidY,
    TidZ,
    NTidX,
    NTidY,
    NTidZ,
    CtaidX,
    CtaidY,
    CtaidZ,
    NCtaidX,
    NCtaidY,
    NCtaidZ,
    Warpsize,
}

impl Intrinsic {
    fn name(&self) -> &'static str {
        match self {
            Intrinsic::TidX => "llvm.nvvm.read.ptx.sreg.tid.x",
            Intrinsic::TidY => "llvm.nvvm.read.ptx.sreg.tid.y",
            Intrinsic::TidZ => "llvm.nvvm.read.ptx.sreg.tid.z",
            Intrinsic::NTidX => "llvm.nvvm.read.ptx.sreg.ntid.x",
            Intrinsic::NTidY => "llvm.nvvm.read.ptx.sreg.ntid.y",
            Intrinsic::NTidZ => "llvm.nvvm.read.ptx.sreg.ntid.z",
            Intrinsic::CtaidX => "llvm.nvvm.read.ptx.sreg.ctaid.x",
            Intrinsic::CtaidY => "llvm.nvvm.read.ptx.sreg.ctaid.y",
            Intrinsic::CtaidZ => "llvm.nvvm.read.ptx.sreg.ctaid.z",
            Intrinsic::NCtaidX => "llvm.nvvm.read.ptx.sreg.nctaid.x",
            Intrinsic::NCtaidY => "llvm.nvvm.read.ptx.sreg.nctaid.y",
            Intrinsic::NCtaidZ => "llvm.nvvm.read.ptx.sreg.nctaid.z",
            Intrinsic::Warpsize => "llvm.nvvm.read.ptx.sreg.warpsize",
        }
    }

    fn code_gen<'ctx>(&self, code_gen: &CodeGen<'ctx>) -> FunctionValue<'ctx> {
        let args = &[];
        let ft = code_gen.context.i32_type().fn_type(args, false);
        let func = code_gen.module.add_function(self.name(), ft, Some(Linkage::External));
        func
    }

    fn f64_call<'ctx>(&self, code_gen: &CodeGen<'ctx>) -> BasicMetadataValueEnum<'ctx> {
        let func = code_gen.module.get_function(self.name())
            .unwrap_or(self.code_gen(code_gen));
        let tmp = code_gen.builder.build_call(func, &[], self.name());
        let res = tmp.try_as_basic_value().left().unwrap().into_int_value();
        code_gen.builder.build_signed_int_to_float(res, code_gen.context.f64_type(), "instr").into()
    }
}

#[derive(Clone, Copy, Debug, PartialOrd, PartialEq)]
pub enum Binop {
    Add,
    Sub,
    Mul,
    Div,
    Lsr,
    Gtr,
    Eq,
    Leq,
    Geq,
    And,
    Or,
    Xor,
    Neq,
    Assign,
}

impl Binop {
    /// Returns the operators precedence
    fn precedence(&self) -> usize {
        match self {
            Self::Assign => 10,
            // --
            Self::Lsr => 20,
            Self::Gtr => 20,
            Self::Eq => 20,
            Self::Leq => 20,
            Self::Geq => 20,
            Self::Neq => 20,
            // --
            Self::Add => 40,
            Self::Sub => 40,
            // --
            Self::Mul => 60,
            Self::Div => 60,
            // --
            Self::And => 80,
            Self::Or => 80,
            Self::Xor => 80,
        }
    }
}

impl TryFrom<Punct> for Binop {
    type Error = ();

    fn try_from(value: Punct) -> Result<Self, Self::Error> {
        match value {
            Punct::Plus => Ok(Binop::Add),
            Punct::Minus => Ok(Binop::Sub),
            Punct::Astrix => Ok(Binop::Mul),
            Punct::Slash => Ok(Binop::Div),
            Punct::Lsr => Ok(Binop::Lsr),
            Punct::Gtr => Ok(Binop::Gtr),
            Punct::Eq => Ok(Binop::Eq),
            Punct::Leq => Ok(Binop::Leq),
            Punct::Geq => Ok(Binop::Geq),
            Punct::And => Ok(Binop::And),
            Punct::Or => Ok(Binop::Or),
            Punct::Xor => Ok(Binop::Xor),
            Punct::Neq => Ok(Binop::Neq),
            Punct::Assign => Ok(Binop::Assign),
            _ => Err(())
        }
    }
}

#[derive(Clone, Debug)]
pub enum PreExpr {
    Ref(Box<AstExpr>),
    Deref(Box<AstExpr>),
    Inv(Box<AstExpr>),
    Unary(Box<AstExpr>),
}

#[derive(Clone, Debug)]
pub enum AstExpr {
    /// Literal expression (only floating-point numbers for this lang)
    Lit(f64),
    /// Variables
    Var(String),
    /// Binary expression
    /// # Parameters
    /// 1. Operator
    /// 2. LHS
    /// 3. RHS
    Bin(Binop, Box<AstExpr>, Box<AstExpr>),
    /// Call expression (function call)
    Call(String, Vec<AstExpr>),
    Pre(PreExpr),
    Field(Box<AstExpr>, String),
    Index(Box<AstExpr>, Box<AstExpr>),
    If(Box<AstExpr>, Box<AstExpr>, Box<AstExpr>),
    For(String, Box<AstExpr>, Box<AstExpr>, Box<AstExpr>, Box<AstExpr>)
}

pub struct CodeGen<'ctx> {
    context: &'ctx Context,
    module: Module<'ctx>,
    builder: Builder<'ctx>,
    named_values: HashMap<String, BasicMetadataValueEnum<'ctx>>,
    fpm: PassManager<FunctionValue<'ctx>>,
}

impl AstExpr {
    pub fn code_gen<'ctx>(&self, code_gen: &mut CodeGen<'ctx>) -> BasicMetadataValueEnum<'ctx> {
        match self {
            Self::Lit(val) => code_gen.context.f64_type().const_float(*val).into(),
            Self::Var(name) => *code_gen.named_values.get(name)
                .expect(&format!("var with name '{name}' not found!")),
            Self::Field(lhs, field) => {
                if let AstExpr::Var(n) = lhs.as_ref().clone() {
                    match (n, field) {
                        (n, field) if n == "threadIdx" && field == "x" => Intrinsic::TidX,
                        (n, field) if n == "threadIdx" && field == "y" => Intrinsic::TidY,
                        (n, field) if n == "threadIdx" && field == "z" => Intrinsic::TidZ,
                        (n, field) if n == "blockIdx" && field == "x" => Intrinsic::CtaidX,
                        (n, field) if n == "blockIdx" && field == "y" => Intrinsic::CtaidY,
                        (n, field) if n == "blockIdx" && field == "z" => Intrinsic::CtaidZ,
                        (n, field) if n == "blockDim" && field == "x" => Intrinsic::NTidX,
                        (n, field) if n == "blockDim" && field == "y" => Intrinsic::NTidY,
                        (n, field) if n == "blockDim" && field == "z" => Intrinsic::NTidZ,
                        (n, field) if n == "gridDim" && field == "x" => Intrinsic::NCtaidX,
                        (n, field) if n == "gridDim" && field == "y" => Intrinsic::NCtaidY,
                        (n, field) if n == "gridDim" && field == "z" => Intrinsic::NCtaidZ,
                        _ => todo!(),
                    }.f64_call(code_gen)
                } else {
                    panic!("fields can only be attached to variables for now")
                }
            },
            Self::Bin(op, lhs, rhs) => {
                let l = lhs.code_gen(code_gen);
                let r = rhs.code_gen(code_gen);

                // check for pointers
                if l.is_float_value() && r.is_float_value() {
                    // treat as float operands
                    let l = l.into_float_value();
                    let r = r.into_float_value();
                    match op {
                        Binop::Add => code_gen.builder.build_float_add(l, r, "addtmp").into(),
                        Binop::Sub => code_gen.builder.build_float_sub(l, r, "subtmp").into(),
                        Binop::Mul => code_gen.builder.build_float_mul(l, r, "multmp").into(),
                        Binop::Div => code_gen.builder.build_float_div(l, r, "divtmp").into(),
                        Binop::Lsr => {  // ordinary less then `l < r`
                            let tmp = code_gen.builder.build_float_compare(FloatPredicate::OLT, l, r, "lsrtmp");
                            code_gen.builder.build_unsigned_int_to_float(tmp, code_gen.context.f64_type(), "booltmp").into()
                        },
                        Binop::Gtr => {  // ordinary less then `l > r`
                            let tmp = code_gen.builder.build_float_compare(FloatPredicate::OGT, l, r, "gtrtmp");
                            code_gen.builder.build_unsigned_int_to_float(tmp, code_gen.context.f64_type(), "booltmp").into()
                        },
                        Binop::Eq => {  // ordinary equals `l == r`
                            let tmp = code_gen.builder.build_float_compare(FloatPredicate::OEQ, l, r, "eqtmp");
                            code_gen.builder.build_unsigned_int_to_float(tmp, code_gen.context.f64_type(), "booltmp").into()
                        },
                        Binop::Neq => {  // ordinary not equals `l != r`
                            let tmp = code_gen.builder.build_float_compare(FloatPredicate::ONE, l, r, "neqtmp");
                            code_gen.builder.build_unsigned_int_to_float(tmp, code_gen.context.f64_type(), "booltmp").into()
                        },
                        Binop::Geq => {  // ordinary greater equals `l >= r`
                            let tmp = code_gen.builder.build_float_compare(FloatPredicate::OGE, l, r, "geqtmp");
                            code_gen.builder.build_unsigned_int_to_float(tmp, code_gen.context.f64_type(), "booltmp").into()
                        },
                        Binop::Leq => {  // ordinary lesser equals `l <= r`
                            let tmp = code_gen.builder.build_float_compare(FloatPredicate::OLE, l, r, "leqtmp");
                            code_gen.builder.build_unsigned_int_to_float(tmp, code_gen.context.f64_type(), "booltmp").into()
                        },
                        _ => unimplemented!()
                    }
                } else if l.is_pointer_value() {
                    if !r.is_float_value() {
                        panic!("lhs is a pointer, rhs is expected to be a number value");
                    }
                    let l = l.into_pointer_value();
                    let r = r.into_float_value();
                    if *op == Binop::Assign {
                        // assign and return the assigned data
                        code_gen.builder.build_store(l, r);
                        r.into()
                    } else {
                        // cast r into an integer number
                        let ptr_ty = l.get_type();
                        let r = code_gen.builder.build_float_to_signed_int(r, code_gen.context.i64_type(), "tmpoff");
                        let l = code_gen.builder.build_ptr_to_int(l, code_gen.context.i64_type(), "tmpptr");
                        let res = match op {
                            Binop::Add => code_gen.builder.build_int_add(r, l, "tmpptrmath"),
                            Binop::Sub => code_gen.builder.build_int_sub(r, l, "tmpptrmath"),
                            _ => unimplemented!(),
                        };
                        code_gen.builder.build_int_to_ptr(res, ptr_ty, "ptradd").into()
                    }
                } else {
                    // r must be a pointer value
                    if !r.is_pointer_value() || !l.is_float_value() {
                        panic!("lhs must be a number value and rhs is expected to be a pointer");
                    }
                    let l = l.into_float_value();
                    let r = r.into_pointer_value();
                    // cast r into an integer number
                    let ptr_ty = r.get_type();
                    let l = code_gen.builder.build_float_to_signed_int(l, code_gen.context.i64_type(), "tmpoff");
                    let r = code_gen.builder.build_ptr_to_int(r, code_gen.context.i64_type(), "tmpptr");
                    let res = match op {
                        Binop::Add => code_gen.builder.build_int_add(r, l, "tmpptrmath"),
                        Binop::Sub => code_gen.builder.build_int_sub(r, l, "tmpptrmath"),
                        _ => unimplemented!(),
                    };
                    code_gen.builder.build_int_to_ptr(res, ptr_ty, "ptradd").into()
                }
            },
            Self::Pre(op) => {
                match op {
                    PreExpr::Ref(expr) => {
                        let val = expr.code_gen(code_gen);
                        assert!(val.is_float_value(), "only numeral values can be referenced");
                        val.into_pointer_value().into()
                    }
                    PreExpr::Deref(expr) => {
                        let val = expr.code_gen(code_gen);
                        assert!(val.is_pointer_value(), "only pointer values can be de-referenced");
                        let ptr = val.into_pointer_value();
                        let pointee_ty = code_gen.context.f64_type();

                        // ignore intellij error, false positive
                        let load = code_gen.builder.build_load(pointee_ty, ptr, "tmpload");
                        load.into()
                    }
                    PreExpr::Inv(expr) => {
                        let val = expr.code_gen(code_gen);
                        assert!(val.is_float_value(), "only numeral values can be de-referenced");
                        let val = val.into_float_value();

                        let bool_val = code_gen.builder.build_bitcast(val, code_gen.context.bool_type(), "tmpbool");
                        let not = code_gen.builder.build_not(bool_val.into_int_value(), "tmpnot");
                        code_gen.builder.build_unsigned_int_to_float(not, code_gen.context.f64_type(), "tmpfloatnot").into()
                    }
                    PreExpr::Unary(expr) => {
                        let val = expr.code_gen(code_gen);
                        assert!(val.is_float_value(), "the unary operator can only be applied to numeral values");

                        let val = val.into_float_value();
                        code_gen.builder.build_float_neg(val, "tmpunary").into()
                    }
                }
            }
            Self::Call(func, args) => {
                let func = code_gen.module.get_function(func)
                    .expect(&format!("unknown function name {func}"));

                if func.count_params() != args.len() as u32 {
                    panic!("Wrong number of arguments");
                }

                let mut params = vec![];
                for a in args {
                    params.push(a.code_gen(code_gen));
                }
                let val = code_gen.builder.build_call(func, &params, "calltmp");
                let base: BasicValueEnum = val.try_as_basic_value().left().unwrap();
                base.into()
            }
            Self::Index(lhs, rhs) => {
                let l = lhs.code_gen(code_gen).into_pointer_value();
                let r = rhs.code_gen(code_gen).into_float_value();

                let idx = code_gen.builder.build_float_to_unsigned_int(r, code_gen.context.i64_type(), "tmpidx");
                unsafe {
                    // ignore intellij error, false positive
                    code_gen.builder.build_gep(
                        code_gen.context.f64_type(),
                        l,
                        &[idx],
                        "tmpgep"
                    ).into()
                }
            },
            Self::If(cond, then, el) => {
                let cond = cond.code_gen(code_gen);
                assert!(cond.is_float_value(), "conditional values must be numerals");

                let ifcond = code_gen.builder.build_float_compare(
                    FloatPredicate::ONE,
                    cond.into_float_value(),
                    code_gen.context.f64_type().const_zero(), "ifcond");
                let func = code_gen.builder.get_insert_block().unwrap()
                    .get_parent().unwrap();
                // create blocks
                let mut then_bb = code_gen.context.append_basic_block(func, "then");
                let mut else_bb = code_gen.context.insert_basic_block_after(then_bb, "else");
                let merge_bb = code_gen.context.insert_basic_block_after(else_bb, "ifcont");

                code_gen.builder.build_conditional_branch(ifcond, then_bb, else_bb);
                // emit into then block
                code_gen.builder.position_at_end(then_bb);
                let then_value: BasicValueEnum = then.code_gen(code_gen).try_into().unwrap();
                code_gen.builder.build_unconditional_branch(merge_bb);
                then_bb = code_gen.builder.get_insert_block().unwrap();

                // emit else block
                code_gen.builder.position_at_end(else_bb);
                let else_value: BasicValueEnum = el.code_gen(code_gen).try_into().unwrap();
                code_gen.builder.build_unconditional_branch(merge_bb);
                else_bb = code_gen.builder.get_insert_block().unwrap();

                // emit merge block
                code_gen.builder.position_at_end(merge_bb);
                let phi_node = code_gen.builder.build_phi(code_gen.context.f64_type(), "iftmp");
                phi_node.add_incoming(&[
                    (&then_value, then_bb),
                    (&else_value, else_bb),
                ]);
                phi_node.as_basic_value().into()
            },
            Self::For(iter, start, end, step, body) => {
                // emit the start code first, without 'variable' in scope
                let start_value: BasicValueEnum = start.code_gen(code_gen).try_into().unwrap();
                // make the new block for the loop header, inserting after the current block
                let func = code_gen.builder.get_insert_block().unwrap()
                    .get_parent().unwrap();
                let preheader_bb = code_gen.builder.get_insert_block().unwrap();
                let loop_bb = code_gen.context.append_basic_block(func, "loop");

                // insert an explicit fall through from the current block to the loop bb
                code_gen.builder.build_unconditional_branch(loop_bb);
                code_gen.builder.position_at_end(loop_bb);
                // start the phi node with an entry for start
                let iterator = code_gen.builder.build_phi(code_gen.context.f64_type(), iter);
                iterator.add_incoming(&[(&start_value, preheader_bb)]);
                // within the loop, the variable is defined equal to the phi node. If it
                // shadows an existing variable, we have to restore it, so save it now.
                let old_value = code_gen.named_values.remove(iter);
                code_gen.named_values.insert(iter.to_owned(), iterator.as_basic_value().into());

                // emit the body of the loop. This, like any other expr, can change the current bb.
                // note that we ignore the value computed by the body, but don't allow an error.
                body.code_gen(code_gen);
                // emit step value
                let step_value = step.code_gen(code_gen).into_float_value();
                let next_iterator = code_gen.builder.build_float_add(
                    iterator.as_basic_value().into_float_value(),
                    step_value, "nextvar");
                // compute end condition
                let end_cond = end.code_gen(code_gen).into_float_value();
                let end_cond = code_gen.builder.build_float_compare(
                    FloatPredicate::ONE,
                    end_cond,
                    code_gen.context.f64_type().const_zero(),
                    "loopcond");

                // create the 'after loop' block and insert it
                let loop_end_bb = code_gen.builder.get_insert_block().unwrap();
                let after_bb = code_gen.context.append_basic_block(
                    func, "afterloop");
                // insert the conditional branch into the end of loop_end_bb
                code_gen.builder.build_conditional_branch(end_cond, loop_bb, after_bb);
                code_gen.builder.position_at_end(after_bb);

                // add a new entry to the phi node for the backedge
                iterator.add_incoming(&[(&next_iterator, loop_end_bb)]);
                // restore named variable from before
                if let Some(var) = old_value {
                    code_gen.named_values.remove(iter);
                    code_gen.named_values.insert(iter.to_owned(), var);
                }
                // always return 0.0
                code_gen.context.f64_type().const_zero().into()
            },
        }
    }
}

/// a function argument
#[derive(Debug)]
enum ArgumentAst {
    /// an argument passed by value
    Val(String),
    /// an argument passed by pointer
    Ptr(String),
}

#[derive(Debug)]
pub struct PrototypeAst {
    name: String,
    args: Vec<ArgumentAst>,
    is_global: bool,
    is_const: bool,
}

impl ArgumentAst {
    fn name(&self) -> &str {
        match &self {
            ArgumentAst::Val(name) => name,
            ArgumentAst::Ptr(name) => name,
        }
    }
}

impl PrototypeAst {
    pub fn code_gen<'ctx>(&self, code_gen: &CodeGen<'ctx>) -> FunctionValue<'ctx> {
        let args: Vec<_> = self.args
            .iter().map(|ty| {

            match ty {
                ArgumentAst::Val(_) => code_gen.context.f64_type().into(),
                ArgumentAst::Ptr(_) => code_gen.context.f64_type().ptr_type(AddressSpace::from(1)).into(),
            }
        })
            .collect();
        let ft = if self.is_global {
            code_gen.context.void_type().fn_type(&args, false)
        } else {
            code_gen.context.f64_type().fn_type(&args, false)
        };
        let func = code_gen.module.add_function(&self.name, ft, Some(Linkage::External));

        if self.is_global {
            let global_func = func.as_global_value();
            let kernel_annotation: BasicMetadataValueEnum = code_gen.context.metadata_string("kernel").into();
            let data = code_gen.context.metadata_node(&[
                global_func.as_pointer_value().into(),
                kernel_annotation,
                code_gen.context.i32_type().const_int(1, false).into(),
            ]);
            code_gen.module.add_global_metadata("nvvm.annotations", &data)
                .unwrap();
        }
        if self.is_const {
            let attrib_kind = Attribute::get_named_enum_kind_id("readnone");
            func.add_attribute(AttributeLoc::Function, code_gen.context.create_enum_attribute(attrib_kind, 0));
        }

         // set names for all args
        for (n, arg) in self.args.iter().enumerate() {
            let param = func.get_nth_param(n as u32).unwrap();
            match &arg {
                ArgumentAst::Val(name) => {
                    param.set_name(name);
                },
                ArgumentAst::Ptr(name) => {
                    param.set_name(name);
                    // func.add_attribute(
                    //     AttributeLoc::Param(n as u32),
                    //     code_gen.context.create_enum_attribute(attrib_kid, 1)
                    // );
                },
            };
        }
        func
    }
}

#[derive(Debug)]
pub struct FunctionAst {
    proto: PrototypeAst,
    body: AstExpr,
}

impl FunctionAst {
    pub fn code_gen<'ctx>(&self, code_gen: &mut CodeGen<'ctx>) -> FunctionValue<'ctx> {
        let func = code_gen.module.get_function(&self.proto.name)
            .unwrap_or(self.proto.code_gen(code_gen));

        if func.is_undef() {
            panic!("Cannot infer function '{}'", self.proto.name);
        }
        // create a new basic block to start insertion into
        let block = code_gen.context.append_basic_block(func, "entry");
        code_gen.builder.position_at_end(block);

        // record the function arguments in the named values map
        for n in 0..func.count_params() {
            let arg = func.get_nth_param(n).unwrap();
            code_gen.named_values.insert(self.proto.args[n as usize].name().to_owned(), arg.into());
        }

        // build body
        let body = self.body.code_gen(code_gen).into_float_value();
        if self.proto.is_global {
            code_gen.builder.build_return(None); // return void
        } else {
            code_gen.builder.build_return(Some(&body));
        }
        code_gen.named_values.clear();

        // verify generated code, checking for consistency
        func.verify(true);
        code_gen.fpm.run_on(&func);
        func
    }
}

pub struct Parser<'a> {
    lexer: Lexer<'a>,
    cur_tok: Token,
}

impl<'a> Parser<'a> {
    pub fn new(mut lex: Lexer<'a>) -> Result<Self, LexError> {
        Ok(Parser {
            cur_tok: lex.get_tok()?,
            lexer: lex
        })
    }

    /// Tries to get the next available token from the lexer. Any errors generated during lexing
    /// are passed down to this function.
    fn next_token(&mut self) -> Result<&Token, LexError> {
        self.cur_tok = self.lexer.get_tok()?;
        Ok(&self.cur_tok)
    }

    pub fn exec(&mut self) -> Result<(), LexError> {
        // init codegen stuff
        let context = Context::create();
        let module = context.create_module("culeidoscope");
        let builder = context.create_builder();

        // optimize
        let fpm: PassManager<FunctionValue> = PassManager::create(&module);
        // fpm.initialize();
        fpm.add_instruction_combining_pass();
        fpm.add_reassociate_pass();
        fpm.add_gvn_pass();
        fpm.add_cfg_simplification_pass();
        fpm.add_aggressive_inst_combiner_pass();
        fpm.initialize();
        // fpm.finalize();

        let mut code_gen = CodeGen {
            context: &context,
            builder,
            module,
            named_values: Default::default(),
            fpm,
        };


        loop {
            match self.cur_tok {
                Token::Eof => { break; },
                Token::Punct(Punct::Semicolon) => { self.next_token()?; }, // ignore top-level semis
                Token::Def => {
                    // handle def
                    let def = self.parse_def()?;
                    println!("parsed definition {def:?}");
                    // code gen
                    let code = def.code_gen(&mut code_gen);
                    println!("code:");
                    code.print_to_stderr();
                },
                Token::Extern => {
                    // handle extern
                    let ext = self.parse_extern()?;
                    println!("parsed extern {ext:?}");
                    // code gen
                    let code = ext.code_gen(&code_gen);
                    println!("code:");
                    code.print_to_stderr();
                },
                _ => {
                    // handle top level expression
                    let expr = self.parse_primary()?;
                    println!("parsed top-level expression {expr:?}");
                    // code gen
                    let code = expr.code_gen(&mut code_gen);
                    println!("code:{}", code.print_to_string());
                }
            }
        }
        // code_gen.fpm.finalize();



        println!("module:\n\n");
        code_gen.module.print_to_file(Path::new("test.ir")).
            expect("Failed to write LLVM-IR to file");
        println!("\n\n");

        let init_config = InitializationConfig {
            .. Default::default()
        };
        Target::initialize_all(&init_config);

        // nvptx64-nvidia-cuda
        // x86_64-unknown-linux-gnu
        let triple = TargetTriple::create("nvptx64-nvidia-cuda");
        let target = Target::from_triple(&triple)
            .expect("Failed to initialize nvptx target");

        // env::set_var("CUDACXX", "/usr/local/cuda/bin");

        println!("target: {:?}", target.get_description());
        println!("target name: {:?}", target.get_name());
        println!("has target machine: {}", target.has_target_machine());
        println!("has asm backend: {}", target.has_asm_backend());
        println!("has jit: {}", target.has_jit());

        code_gen.module.set_triple(&triple);
        let machine = target.create_target_machine(
            &triple, "sm_52", "", OptimizationLevel::Default, RelocMode::Default, CodeModel::Default
        ).expect("Failed to create target machine");
        code_gen.module.set_data_layout(&machine.get_target_data().get_data_layout());

        // optimize
        let pm_builder = PassManagerBuilder::create();
        pm_builder.set_optimization_level(OptimizationLevel::Aggressive);
        let fpm = PassManager::create(());
        pm_builder.populate_module_pass_manager(&fpm);
        machine.add_analysis_passes(&fpm);

        let fpm: PassManager<FunctionValue> = PassManager::create(&code_gen.module);
        fpm.add_instruction_combining_pass();
        fpm.add_reassociate_pass();
        fpm.add_gvn_pass();
        fpm.add_cfg_simplification_pass();
        fpm.initialize();
        machine.add_analysis_passes(&fpm);
        fpm.finalize();

        // emit code
        machine.write_to_file(&code_gen.module, FileType::Assembly, Path::new("test.ptx"))
            .expect("Failed to write to file");
        Ok(())
    }

    #[allow(dead_code)]
    pub fn parse_top_level_expr(&mut self) -> Result<FunctionAst, LexError> {
        let expr = self.parse_primary()?;
        let proto = PrototypeAst {
            name: "".to_owned(),
            args: vec![],
            is_global: false,
            is_const: false,
        };
        Ok(FunctionAst {
            proto,
            body: expr
        })
    }

    pub fn parse_extern(&mut self) -> Result<PrototypeAst, LexError> {
        if self.cur_tok != Token::Extern {
            return Err(self.expect_token(Token::Extern));
        }
        self.next_token()?;
        self.parse_proto()
    }

    pub fn parse_def(&mut self) -> Result<FunctionAst, LexError> {
        if self.cur_tok != Token::Def {
            return Err(self.expect_token(Token::Def));
        }
        self.next_token()?;
        // parse prototype
        let proto = self.parse_proto()?;
        Ok(FunctionAst {
            proto,
            body: self.parse_primary()?,
        })
    }

    pub fn parse_proto(&mut self) -> Result<PrototypeAst, LexError> {
        let is_global = if self.cur_tok == Token::Global {
            self.next_token()?;
            true
        } else {
            false
        };
        let is_const = if self.cur_tok == Token::Const {
            self.next_token()?;
            true
        } else {
            false
        };

        let name = if let Token::Ident(ident) = self.cur_tok.clone() {
            Ok(ident)
        } else {
            Err(self.expect_token(Token::Ident("fn_name".to_owned())))
        }?;
        self.next_token()?;

        // expect opening bracket
        if self.cur_tok != Token::Punct(Punct::BracketOpen) {
            return Err(self.expect_token(Token::Punct(Punct::BracketOpen)));
        }
        self.next_token()?;

        // parse list of args
        let mut args = Vec::new();
        while let Token::Ident(name) = self.cur_tok.clone() {
            self.next_token()?;
            args.push(if Token::Punct(Punct::Astrix) == self.cur_tok {
                // this is a pointer
                self.next_token()?;
                ArgumentAst::Ptr(name)
            } else {
                // this is a value
                ArgumentAst::Val(name)
            });
        }

        // expect closing bracket
        if self.cur_tok != Token::Punct(Punct::BracketClose) {
            return Err(self.expect_token(Token::Punct(Punct::BracketClose)));
        }
        self.next_token()?;

        Ok(PrototypeAst {
            name,
            args,
            is_global,
            is_const,
        })
    }

    fn parse_if(&mut self) -> Result<AstExpr, LexError> {
        if self.cur_tok != Token::If {
            return Err(self.expect_token(Token::If));
        }
        self.next_token()?;

        // condition
        let cond = self.parse_primary()?;
        if self.cur_tok != Token::Then {
            return Err(self.expect_token(Token::Then));
        }
        self.next_token()?;

        // parse then
        let then = self.parse_primary()?;
        let el = if self.cur_tok == Token::Else {
            self.next_token()?;
            Box::new(self.parse_primary()?)
        } else {
            return Err(self.expect_token(Token::Else));
        };

        Ok(AstExpr::If(
            Box::new(cond),
            Box::new(then),
            el)
        )
    }

    fn parse_for(&mut self) -> Result<AstExpr, LexError> {
        if self.cur_tok != Token::For {
            return Err(self.expect_token(Token::For));
        }
        self.next_token()?;

        // read iterator
        let name = if let Token::Ident(name) = &self.cur_tok {
            name.clone()
        } else {
            return Err(self.expect_token(Token::Ident("iterator".to_owned())));
        };
        self.next_token()?;

        // eat `in`
        if self.cur_tok != Token::Punct(Punct::Assign) {
            return Err(self.expect_token(Token::Punct(Punct::Assign)));
        }
        self.next_token()?;

        let start = self.parse_primary()?;
        if self.cur_tok != Token::Punct(Punct::Comma) {
            return Err(self.expect_token(Token::Punct(Punct::Comma)));
        }
        self.next_token()?;
        let end = self.parse_primary()?;

        // optional step value
        let step = if self.cur_tok == Token::Punct(Punct::Comma) {
            self.next_token()?;
            self.parse_primary()?
        } else {
            AstExpr::Lit(1.0)
        };
        // eat `in`
        if self.cur_tok != Token::In {
            return Err(self.expect_token(Token::In));
        }
        self.next_token()?;

        let body = self.parse_primary()?;
        Ok(AstExpr::For(name, Box::new(start), Box::new(end), Box::new(step), Box::new(body)))
    }

    pub fn parse_expr(&mut self) -> Result<AstExpr, LexError> {
        let mut expr = match self.cur_tok.clone() {
            Token::Number(num) => {
                self.next_token()?;
                AstExpr::Lit(num)
            },
            Token::Punct(Punct::BracketOpen) => {
                self.next_token()?;
                let inner = self.parse_primary()?;
                if self.cur_tok != Token::Punct(Punct::BracketClose) {
                    // expect the parenthesis to close
                    return Err(LexError::WrongTokenType(
                        self.lexer.pos, self.cur_tok.clone(), Token::Punct(Punct::BracketClose)));
                }
                self.next_token()?;
                inner
            },
            Token::Ident(name) => {
                self.next_token()?;
                if self.cur_tok == Token::Punct(Punct::BracketOpen) {
                    // function call
                    self.next_token()?;
                    let mut args = Vec::new();
                    loop {
                        if self.cur_tok == Token::Punct(Punct::BracketClose) {
                            self.next_token()?;
                            break AstExpr::Call(name, args);
                        }
                        args.push(self.parse_primary()?);
                        if self.cur_tok != Token::Punct(Punct::Comma) {
                            return Err(LexError::WrongTokenType(
                                self.lexer.pos, self.cur_tok.clone(), Token::Punct(Punct::Comma)
                            ));
                        }
                        self.next_token()?; // eat `,`
                    }
                } else {
                    // simple variable name
                    AstExpr::Var(name)
                }
            },
            Token::If => {
                self.parse_if()?
            },
            Token::For => {
                self.parse_for()?
            },
            Token::Punct(Punct::And) => {
                self.next_token()?;
                AstExpr::Pre(PreExpr::Ref(Box::new(self.parse_expr()?)))
            },
            Token::Punct(Punct::Astrix) => {
                self.next_token()?;
                AstExpr::Pre(PreExpr::Deref(Box::new(self.parse_expr()?)))
            },
            Token::Punct(Punct::Not) => {
                self.next_token()?;
                AstExpr::Pre(PreExpr::Inv(Box::new(self.parse_expr()?)))
            },
            Token::Punct(Punct::Plus) => {
                self.next_token()?;
                self.parse_expr()?
            },
            Token::Punct(Punct::Minus) => {
                self.next_token()?;
                AstExpr::Pre(PreExpr::Unary(Box::new(self.parse_expr()?)))
            },
            tt => {
                println!("token '{:?}' is not implemented", tt);
                unimplemented!()
            }
        };

        // parse field
        if self.cur_tok == Token::Punct(Punct::Dot) {
            self.next_token()?;
            let name = if let Token::Ident(name) = self.cur_tok.clone() {
                Ok(name)
            } else {
                Err(self.expect_token(Token::Ident("field".to_owned())))
            }?;
            self.next_token()?;
            expr = AstExpr::Field(Box::new(expr), name);
        }
        // parse index
        if self.cur_tok == Token::Punct(Punct::IndexOpen) {
            self.next_token()?;
            let idx = self.parse_primary()?;

            if self.cur_tok != Token::Punct(Punct::IndexClose) {
                return Err(self.expect_token(Token::Punct(Punct::IndexClose)));
            }
            self.next_token()?;
            expr = AstExpr::Index(Box::new(expr), Box::new(idx));
        }
        Ok(expr)
    }

    /// Tries to parse the current token as an expression.
    pub fn parse_primary(&mut self) -> Result<AstExpr, LexError> {
        let mut expr = self.parse_expr()?;
        // try to parse binop
        expr = self.parse_binop(0, expr)?;
        Ok(expr)
    }

    /// Gets the precedence of the current token, if possible. If the current token is not an
    /// operator with a precedence value, this function return `None`.
    fn token_precedence(&self) -> Option<usize> {
        if let Token::Punct(p) = self.cur_tok.clone() {
            Binop::try_from(p)
                .ok()
                .map(|op| op.precedence())
        } else {
            None
        }
    }

    fn parse_binop(&mut self, expr_prec: usize, mut lhs: AstExpr) -> Result<AstExpr, LexError> {
        loop {
            let binop = if let Token::Punct(c) = self.cur_tok.clone() {
                if let Ok(binop) = Binop::try_from(c) {
                    binop
                } else {
                    // this is not a binop
                    return Ok(lhs);
                }
            } else {
                return Ok(lhs);
            };

            // if this is a binop that binds at least as tightly as the current binop,
            // consume it, otherwise we are done
            if binop.precedence() < expr_prec {
                return Ok(lhs);
            }
            self.next_token()?;

            // parse expression after the binary operator
            let mut rhs = self.parse_expr()?;
            // if binop binds less tightly with RHS than the operator after RHS, let the pending
            // operator take RHS as its LHS
            if let Some(next) = self.token_precedence() {
                if binop.precedence() < next {
                    rhs = self.parse_binop(binop.precedence() + 1, rhs)?;
                }
            }
            // merge
            lhs = AstExpr::Bin(binop, Box::new(lhs), Box::new(rhs));
        }
    }

    fn expect_token(&self, tt: Token) -> LexError {
        LexError::WrongTokenType(self.lexer.pos, self.cur_tok.clone(), tt)
    }
}

