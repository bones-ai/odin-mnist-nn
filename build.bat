@echo off
setlocal

:: Check the parameter passed to the script
if "%1" == "size" (
    echo Building for size
    odin build . -o:size -microarch:native -disable-assert -no-bounds-check
) else if "%1" == "speed" (
    echo Building for speed
    odin build . -o:speed -microarch:native -disable-assert -no-bounds-check
) else (
    echo Default build
    odin build .
)

endlocal
