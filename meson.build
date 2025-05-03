project(
  'kaleidoscope',
  ['cpp'],
  meson_version : '>=1.7'
)

cpp = meson.get_compiler('cpp')
sources = [
  'src/main.cpp'
]
deps = [
  dependency('llvm', required: true)
]
cpp_args = ['-std=c++2c', '-Wall', '-Wextra', '-Wshadow', '-Wconversion', '-Wpedantic', '-Werror']
link_args = []

# build targeting the native system this was built on. If buildtype is debug,
# no target architecture will be specified, to ensure the best debugging experience.
executable(
  'kaleidoscope',
  sources,
  include_directories: [include_directories('include')],
  dependencies: deps,
  cpp_args : cpp_args + (get_option('buildtype').startswith('debug') ? [] : ['-march=native']),
  link_args : link_args
)

# this project is mainly for messing around with LLVM to help me figure out
# how to develop my compiler, so there are no tests,
# all testing is manual
