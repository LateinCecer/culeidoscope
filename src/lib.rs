mod lang;

use std::error::Error;
use inkwell::builder::Builder;
use inkwell::context::Context;
use inkwell::execution_engine::{ExecutionEngine, JitFunction};
use inkwell::module::Module;
use inkwell::OptimizationLevel;

type SumFunc = unsafe extern "C" fn(u64, u64, u64) -> u64;

struct CodeGen<'ctx> {
    context: &'ctx Context,
    module: Module<'ctx>,
    builder: Builder<'ctx>,
    execution_engine: ExecutionEngine<'ctx>,
}

impl<'ctx> CodeGen<'ctx> {
    fn jit_compile_sum(&self) -> Option<JitFunction<SumFunc>> {
        let i64_type = self.context.i64_type();
        let fn_type = i64_type.fn_type(&[i64_type.into(), i64_type.into(), i64_type.into()], false);
        let function = self.module.add_function("sum", fn_type, None);
        let basic_block = self.context.append_basic_block(function, "entry");

        self.builder.position_at_end(basic_block);

        let x = function.get_nth_param(0)?.into_int_value();
        let y = function.get_nth_param(1)?.into_int_value();
        let z = function.get_nth_param(2)?.into_int_value();

        let sum = self.builder.build_int_add(x, y, "sum");
        let sum = self.builder.build_int_add(sum, z, "sum");

        self.builder.build_return(Some(&sum));
        unsafe {
            self.execution_engine.get_function("sum").ok()
        }
    }
}

fn exec() -> Result<(), Box<dyn Error>> {
    let context = Context::create();
    let module = context.create_module("sum");
    let execution_engine = module.create_jit_execution_engine(OptimizationLevel::None)?;
    let codegen = CodeGen {
        context: &context,
        module,
        builder: context.create_builder(),
        execution_engine,
    };

    let sum = codegen.jit_compile_sum().ok_or("Unable to JIT Compile `sum`")?;
    let x = 1u64;
    let y = 2u64;
    let z = 3u64;

    unsafe {
        println!("{} + {} + {} = {}", x, y, z, sum.call(x, y, z));
        assert_eq!(sum.call(x, y, z), x + y + z)
    }
    Ok(())
}


pub fn add(left: usize, right: usize) -> usize {
    left + right
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::Path;
    use cust::context::{CacheConfig, ContextFlags, CurrentContext, SharedMemoryConfig};
    use cust::{CudaFlags, launch};
    use cust::error::CudaResult;
    use cust::module::{ModuleJitOption, OptLevel};
    use cust::prelude::{CopyDestination, Device, DeviceBuffer};
    use cust::stream::{Stream, StreamFlags};
    use crate::lang::ast::Parser;
    use crate::lang::Lexer;
    use super::*;

    #[test]
    fn code_gen() -> Result<(), Box<dyn Error>> {
        exec()?;
        Ok(())
    }

    #[test]
    fn it_works() -> CudaResult<()> {
        let src = r#"
        # Returns the index of the current thread within a one-dimensional compute grid
        def const idx()
            threadIdx.x + blockDim.x * blockIdx.x

        # Performs a vectorized multiplication between 'lhs' and 'rhs' and stores the result in 'dst'
        def global foo(lhr* rhs* dst* n)
            if idx() < n then   # ensure the index is in bounds
                for i = 0, i < 2 in   # compute multiple entries per thread to reduce overhead
                    dst[idx() + i * n] = *lhr[idx() + i * n] * *rhs[idx() + i * n]
            else
                0   # result is ignored, since global functions are inherently void functions
        "#;
        let lex = Lexer::new(src.chars());
        let mut parser = Parser::new(lex).unwrap();
        parser.exec().unwrap();

        // init cust
        cust::init(CudaFlags::empty())?;
        let devices = Device::devices()?;
        let mut device = None;
        println!("Listing devices...");
        for d in devices.flatten() {
            let name = d.name()?;
            println!(" - {}", name);
            if device.is_none() {
                device = Some(d);
            }
        }

        let device = device.expect("No cuda device available");
        println!("Selected device {}", device.name()?);
        let ctx = cust::prelude::Context::new(device)?;
        ctx.set_flags(ContextFlags::SCHED_AUTO)?;

        // +++ global mem config +++
        CurrentContext::set_cache_config(CacheConfig::PreferL1)?;
        CurrentContext::set_shared_memory_config(SharedMemoryConfig::EightByteBankSize)?;

        // load kernel from generated ptx
        let file = fs::read_to_string(Path::new("test.ptx"))
            .expect("Failed to locate ptx file");
        let module = cust::prelude::Module::from_ptx(
            file, &[
                ModuleJitOption::DetermineTargetFromContext,
                ModuleJitOption::OptLevel(OptLevel::O4)
            ]
        )?;


        let n = 100; // init 100 items
        let m = 3;
        // init test data
        let lhs_raw: Vec<_> = (0..(n * m)).map(|i| i as f64).collect();
        let rhs_raw: Vec<_> = (0..(n * m)).map(|i| i as f64).collect();
        // init buffers
        let lhs = DeviceBuffer::from_slice(&lhs_raw)?;
        let rhs = DeviceBuffer::from_slice(&rhs_raw)?;
        let dst: DeviceBuffer<f64> = DeviceBuffer::zeroed((n * m) as usize)?;
        // get func
        let func = module.get_function("foo")?;
        let (_, block_size) = func.suggested_launch_configuration(0, 0.into())?;
        let grid_size = (block_size + n - 1) / block_size;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
        // launch kernel
        unsafe {
            launch!(
                func<<<grid_size, block_size, 0, stream>>>(
                    lhs.as_device_ptr(),
                    rhs.as_device_ptr(),
                    dst.as_device_ptr(),
                    n as f64,
                )
            )
        }?;
        stream.synchronize()?;
        // download results
        let mut dst_raw = vec![0_f64; (n * m) as usize];
        dst.copy_to(&mut dst_raw)?;
        // compare results
        for (i, &val) in dst_raw.iter().enumerate() {
            // println!("{i}  >>  {} * {} = {}", lhs_raw[i], rhs_raw[i], val);
            assert_eq!(val, lhs_raw[i] * rhs_raw[i]);
        }
        Ok(())
    }
}
