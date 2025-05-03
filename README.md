# Kaleidoscope LLVM Tutorial
This is just my attempt at going through the LLVM Kaleidoscope tutorial in preparation for writing my
own compiler: https://llvm.org/docs/tutorial/index.html

However, on top of this tutorial, I'm trying to practice using the most up-to-date modern C++26 standards and best practices (at least those supported
by Clang 20) and best practices, whereas the Kaleidoscope tutorial is rather old (but still super helpful for me!).

# Building on Windows
Install msys2 and use only the clang-based environment and clang-based toolchain.

Use the newest version of clang (20 ATTOW) and install
the necessary dependencies:

TODO: Below might be out of date - missing llvm dev dependencies?
```shell
# in CLANG terminal (NOT ucrt64)
pacman -S --needed base-devel \
mingw-w64-clang-x86_64-toolchain
pacman -S \
mingw-w64-clang-x86_64-meson 
```

# Building on Linux
We want to use clang (figure that makes the most sense since this is an LLVM-based project)
and the most modern C++ version possible (C++2c).

On my system (PopOS 22.04), to get this project compiling with a new enough version of clang 
it's not super straightforward...This is mostly as someone not super experienced
with C++ on Linux so these may be wrong or there may be an easier way...

## 1. Install latest clang version 
Install latest clang version (20 ATTOW) using the 
automatic installation script-based approach documented here: https://apt.llvm.org/
```shell
wget https://apt.llvm.org/llvm.sh 
chmod +x llvm.sh 
sudo ./llvm.sh 20
```

## 2. Make meson use clang with libc++
This isn't enough. ATTOW, meson defaults to always using libstdc++ even though LLVM has
their own version of this - libc++. 

That's the one we want to use since we know it will be maximally
compatible with the version of clang we're using (for example `#include <format>`). 

To force meson to use libc++ instead, you need to use `meson setup` with
the `--native-file linux-clang.ini` which passes the necessary flags to clang to use libc++ instead of libstdc++.
```shell
meson setup buildDir --native-file linux-clang.ini
```

## 3. Install libc++
That's seemingly STILL not enough! You'd think that using the auto-installation script would
automatically install the necessary dev dependencies for libc++. ATTOW I think there might be an issue
with the package or the script, so you need to manually install it using apt.
```shell
sudo apt install libc++-20-dev libc++abi-20-dev
```

Now it should work!