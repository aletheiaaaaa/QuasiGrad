cmake_minimum_required(VERSION 3.20)

project(agon
  VERSION 0.1.0
  DESCRIPTION "C++ zeroth-order optimization library with SIMD acceleration and PyTorch-inspired API"
  LANGUAGES CXX
)

# Options
option(AGON_ENABLE_NATIVE "Enable native architecture optimizations (-march=native)" ON)
option(AGON_ENABLE_AVX512 "Enable AVX512 instructions (if available)" ON)
option(AGON_ENABLE_AVX2 "Enable AVX2 instructions (if available)" ON)
option(AGON_ENABLE_SSE4 "Enable SSE4.1 instructions (if available)" ON)

find_package(eve QUIET)
if(NOT eve_FOUND)
  include(FetchContent)
  FetchContent_Declare(
    eve
    GIT_REPOSITORY https://github.com/jfalcou/eve.git
    GIT_TAG v2023.02.15
    EXCLUDE_FROM_ALL
  )
  set(EVE_BUILD_TEST OFF CACHE BOOL "" FORCE)
  set(EVE_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
  set(EVE_BUILD_DOCUMENTATION OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(eve)
  set(AGON_EVE_FETCHED TRUE)
endif()

# Header-only library target
add_library(agon INTERFACE)
add_library(agon::agon ALIAS agon)

# Require C++23
target_compile_features(agon INTERFACE cxx_std_23)

# Include directories
target_include_directories(agon
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

if(AGON_EVE_FETCHED)
  target_include_directories(agon INTERFACE $<BUILD_INTERFACE:${eve_SOURCE_DIR}/include>)
else()
  target_link_libraries(agon INTERFACE eve::eve)
endif()

# Compiler-specific settings
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  # Architecture optimizations
  if(AGON_ENABLE_NATIVE)
    target_compile_options(agon INTERFACE -march=native)
  else()
    # Manual SIMD feature flags
    if(AGON_ENABLE_AVX512)
      include(CheckCXXCompilerFlag)
      check_cxx_compiler_flag("-mavx512f" COMPILER_SUPPORTS_AVX512F)
      if(COMPILER_SUPPORTS_AVX512F)
        target_compile_options(agon INTERFACE -mavx512f)
      endif()
    endif()

    if(AGON_ENABLE_AVX2)
      include(CheckCXXCompilerFlag)
      check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)
      if(COMPILER_SUPPORTS_AVX2)
        target_compile_options(agon INTERFACE -mavx2 -mfma)
      endif()
    endif()

    if(AGON_ENABLE_SSE4)
      include(CheckCXXCompilerFlag)
      check_cxx_compiler_flag("-msse4.1" COMPILER_SUPPORTS_SSE41)
      if(COMPILER_SUPPORTS_SSE41)
        target_compile_options(agon INTERFACE -msse4.1)
      endif()
    endif()
  endif()

elseif(MSVC)
  if(AGON_ENABLE_AVX512)
    target_compile_options(agon INTERFACE /arch:AVX512)
  elseif(AGON_ENABLE_AVX2)
    target_compile_options(agon INTERFACE /arch:AVX2)
  elseif(AGON_ENABLE_SSE4)
    target_compile_options(agon INTERFACE /arch:SSE2)
  endif()
endif()

# Installation
include(GNUInstallDirs)

install(TARGETS agon
  EXPORT agonTargets
)

install(DIRECTORY include/agon
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  FILES_MATCHING PATTERN "*.hpp"
)

if(AGON_EVE_FETCHED)
  install(DIRECTORY ${eve_SOURCE_DIR}/include/eve
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  )
endif()

# Export targets
install(EXPORT agonTargets
  FILE agonTargets.cmake
  NAMESPACE agon::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/agon
)

# Package configuration
include(CMakePackageConfigHelpers)

configure_package_config_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/agonConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/agonConfig.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/agon
)

write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/agonConfigVersion.cmake
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion
)

install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/agonConfig.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/agonConfigVersion.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/agon
)

# Export from build tree
export(EXPORT agonTargets
  FILE ${CMAKE_CURRENT_BINARY_DIR}/agonTargets.cmake
  NAMESPACE agon::
)

# Uninstall target
if(NOT TARGET uninstall)
  file(CONFIGURE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake" CONTENT
[[if(NOT EXISTS "@CMAKE_CURRENT_BINARY_DIR@/install_manifest.txt")
  message(FATAL_ERROR "Cannot find install manifest: @CMAKE_CURRENT_BINARY_DIR@/install_manifest.txt")
endif()

file(READ "@CMAKE_CURRENT_BINARY_DIR@/install_manifest.txt" files)
string(REGEX REPLACE "\n" ";" files "${files}")

foreach(file ${files})
  message(STATUS "Uninstalling $ENV{DESTDIR}${file}")
  if(IS_SYMLINK "$ENV{DESTDIR}${file}" OR EXISTS "$ENV{DESTDIR}${file}")
    execute_process(
      COMMAND "@CMAKE_COMMAND@" -E remove "$ENV{DESTDIR}${file}"
      RESULT_VARIABLE rm_retval
    )
    if(NOT "${rm_retval}" STREQUAL "0")
      message(FATAL_ERROR "Problem when removing $ENV{DESTDIR}${file}")
    endif()
  else()
    message(STATUS "File $ENV{DESTDIR}${file} does not exist.")
  endif()
endforeach()
]] @ONLY)
  add_custom_target(uninstall
    COMMAND "${CMAKE_COMMAND}" -P "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
  )
endif()