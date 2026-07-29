@echo off
rem Builds the CUDA texture-rotation solver to .\cuda_solver.exe
rem
rem The kernel is compiled at runtime by NVRTC, so nvcc is never invoked and there
rem is no nvcc/MSVC version compatibility to satisfy -- this is an ordinary C++
rem program that happens to link cuda.lib and nvrtc.lib.
rem
rem Usage:
rem   build_cuda.bat                          build
rem   build_cuda.bat /Zi /Od                  extra flags are passed to cl
rem   set LODESTONE_CUDA_PATH=...             force a specific CUDA toolkit
setlocal enabledelayedexpansion

set "ROOT=%~dp0"
set "SRC=%ROOT%cuda_solver.cpp"
set "OUT=%ROOT%cuda_solver.exe"

rem ---------------------------------------------------------------------------
rem Locate a CUDA toolkit.
rem
rem CUDA_PATH is deliberately NOT trusted: it commonly points at whichever toolkit
rem was installed last rather than the newest, and an old one (pre-11.8) cannot
rem target Ada at all. Pick the highest version that actually has the headers and
rem import libraries we need.
rem ---------------------------------------------------------------------------
set "CUDA_DIR="
set "BEST=0"

if defined LODESTONE_CUDA_PATH (
    if exist "%LODESTONE_CUDA_PATH%\include\nvrtc.h" (
        set "CUDA_DIR=%LODESTONE_CUDA_PATH%"
    ) else (
        echo warning: LODESTONE_CUDA_PATH is not a usable CUDA toolkit, ignoring
    )
)

if not defined CUDA_DIR (
    set "TKROOT=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
    if exist "!TKROOT!" (
        for /d %%D in ("!TKROOT!\v*") do (
            if exist "%%D\include\nvrtc.h" if exist "%%D\lib\x64\nvrtc.lib" (
                set "VER=%%~nxD"
                set "VER=!VER:v=!.0"
                for /f "tokens=1,2 delims=." %%a in ("!VER!") do set /a "SCORE=%%a*1000+%%b"
                if !SCORE! GTR !BEST! (
                    set "BEST=!SCORE!"
                    set "CUDA_DIR=%%D"
                )
            )
        )
    )
)

if not defined CUDA_DIR (
    echo error: no usable CUDA toolkit found.
    echo        Install the CUDA Toolkit ^(11.8 or newer for Ada/RTX 40-series^),
    echo        or point LODESTONE_CUDA_PATH at one containing include\nvrtc.h.
    exit /b 1
)

if !BEST! NEQ 0 if !BEST! LSS 11008 (
    echo warning: CUDA !BEST! is older than 11.8 and cannot target Ada ^(sm_89^).
)

rem ---------------------------------------------------------------------------
rem Locate MSVC. Skip if cl.exe is already usable, e.g. inside a Developer Prompt.
rem ---------------------------------------------------------------------------
where cl.exe >nul 2>&1
if errorlevel 1 (
    set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
    if not exist "!VSWHERE!" (
        echo error: cl.exe is not on PATH and vswhere.exe was not found.
        echo        Install Visual Studio Build Tools with the C++ workload, or run
        echo        this from a "x64 Native Tools Command Prompt".
        exit /b 1
    )
    for /f "usebackq tokens=*" %%I in (`"!VSWHERE!" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do (
        set "VSPATH=%%I"
    )
    if not defined VSPATH (
        echo error: no Visual Studio installation with the C++ toolset was found.
        exit /b 1
    )
    call "!VSPATH!\VC\Auxiliary\Build\vcvars64.bat" >nul
    if errorlevel 1 (
        echo error: vcvars64.bat failed.
        exit /b 1
    )
)

echo CUDA:     !CUDA_DIR!
for /f "tokens=*" %%C in ('where cl.exe 2^>nul') do (
    echo Compiler: %%C
    goto :got_cl
)
:got_cl

rem ---------------------------------------------------------------------------
rem Build. /wd4267 and /wd4244 silence size_t/int narrowing warnings that MSVC
rem emits for standard-library index arithmetic and are not defects here.
rem ---------------------------------------------------------------------------
cl /nologo /O2 /std:c++17 /EHsc /W3 /wd4267 /wd4244 ^
   /I"!CUDA_DIR!\include" /I"%ROOT%." ^
   "%SRC%" ^
   /Fe:"%OUT%" /Fo:"%ROOT%cuda_solver.obj" ^
   /link /LIBPATH:"!CUDA_DIR!\lib\x64" cuda.lib nvrtc.lib %*
if errorlevel 1 (
    echo.
    echo build failed.
    exit /b 1
)

echo Built:    %OUT%
echo.
echo Note: !CUDA_DIR!\bin must be on PATH at runtime so the matching
echo       nvrtc DLL is found.
exit /b 0
