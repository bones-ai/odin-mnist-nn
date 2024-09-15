#!/bin/bash

# Run commands based on the parameter
case "$1" in
  size)
    echo "Building for size"
    odin build . -o:size -microarch:native -disable-assert -no-bounds-check
    ;;
    
  speed)
    echo "Building for speed"
    odin build . -o:speed -microarch:native -disable-assert -no-bounds-check
    ;;
    
  *)
    echo "Default build"
    odin build .
    ;;
esac