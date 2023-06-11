# Culeidoscope
So, about this project: This project is meant as a training exercise, as a first project with LLVM and as a proof of
concept to compile a simple home-brew language to Nvidia-PTX.


LINUX 🐧
========
____
# Build-Guide
Compiling this project is kinda rough, since building LLVM can be kinda rough.
That being said, a basic installation of LLVM will suffice for this project, so long as the targets `nvptx64`, `nvptx`,
as well as the native target are supported.
To build this project, LLVM-15.x is required.

## Required software
There are several different ways to compile LLVM, however, this is an opinionated guide and I will only present the way I chose to do it.
If you do not agree with the build software choices below, feel free to find another install guide online or use your own expertise.

**Required Software:**
- `CMake`
- `Ninja`
- `Clang`

## Compiling 
This step is dedicated to building LLVM-15.x. If this version of LLVM with the required targets is already installed
and accessible on your system, this step can be skipped.
This part loosely follows the install guide provided on the <a href="https://llvm.org/docs/GettingStarted.html">LLVM Website</a>.
If the instructions provided by LLVM themselves change in the future, their guide should probably be favoured over this install guide.

### Cloning the project
First up, the <a href="https://github.com/llvm/llvm-project">llvm-project</a> should be cloned from the official Github repo:

````bash
git clone https://github.com/llvm/llvm-project.git --branch release/15.x
cd llvm-project
````

Since the LLVMs sources are quite big, this step may take a while, depending on the available internet and disk speed.

### Building from source
In this step, we will build LLVM from sources. After that, the compiled binaries can be installed locally.

First up, we will use CMake to configure the build:
````bash
cmake -S llvm -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
````
The optional flag ``-DCMAKE_INSTALL_PREFIX=/path/to/install/target`` can be used to configure where llvm will be installed later down the line.
The default install path is `/usr/local/`. Should the default path not be available to you (e.g. if you are not an administrator), this parameter should be used.
Otherwise, I would advise against installing LLVM anywhere that is not the default path, since it may be harder for other programs to find LLVM without additional configuration.

During configuration, you should see a list of targets that will be available for compilation later.
These should include `nvptx`. If it does not show, your system may not be compatible.

### Building
To build and test the program, run

````bash
ninja -C build check-llvm
````

Like the initial cloning of the repo, this may take a while.

### Installing
If all relevant tests passed and LLVM has been build successfully, its libraries and binaries may be installed.
To due this using the configured build tool, execute

````bash
ninja -C build install
````

If you are installing LLVM to the default location, or another path that requires administrative privileges, this command must be executed as super-user.


WINDOWS 🪟
==========
____

# Build-Guide
Compiling this project is kinda rough, since building LLVM can be kinda rough (especially on Windows, lol).
That being said, a basic installation of LLVM will suffice for this project, so long as the targets `nvptx64`, `nvptx`,
as well as the native target are supported.
To build this project, LLVM-15.x is required.

## Cloning the repo
First up, the <a href="https://github.com/llvm/llvm-project">llvm-project</a> should be cloned from the official Github repo:

````bash
git clone --config core.autocrlf=false https://github.com/llvm/llvm-project.git
cd llvm-project
````

### Building from source
Just follow the steps on https://llvm.org/docs/GettingStartedVS.html or something idk.

MAC 🍎
======
___
This project compiles to PTX, a format that is only really usable by CUDA which in turn requires an Nvidia Gpu (or 2, or 8...).

These are graphic cards that are not available for Mac (at least that is the case at the time of writing).

You get the point?

Also, I spend all my money on graphics cards and can't afford a Mac.

# Generating LLVM-IR
The LLVM intermediate representation of the source code is generate through LLVM-API functions. Functions exposed to
the host device must be annotated as kernel functions so that the NVPTX Backend can differentiate between `__global__`
and `__device__` code. The <a href="https://llvm.org/docs/NVPTXUsage.html">official LLVM NVPTX</a> backend guide
contains an example of how to mark functions as kernel functions:

````ir
define float @my_fmad(float %x, float %y, float %z) {
  %mul = fmul float %x, %y
  %add = fadd float %mul, %z
  ret float %add
}

define void @my_kernel(float* %ptr) {
  %val = load float, float* %ptr
  %ret = call float @my_fmad(float %val, float %val, float %val)
  store float %ret, float* %ptr
  ret void
}

!nvvm.annotations = !{!1}
!1 = !{void (float*)* @my_kernel, !"kernel", i32 1}
````

The important part here is the code contained within the last two lines of IR:
````ir
!nvvm.annotations = !{!1}
!1 = !{void (float*)* @my_kernel, !"kernel", i32 1}
````

However, there appears to be an error with the online guide. Passing the function reference into the metadata node
using the LLVM-API results in a metadata entry `ptr @my_kernel`, with produces the desired results. Seeing as the syntax
for pointer values in function parameters is also different between the guide and the generated IR, my guess is that
the guide has not been updated to LLVM-15.x.