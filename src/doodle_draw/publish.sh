#!/bin/bash
set -e


TARGET=../../demos/doodle_draw

mkdir -p "$TARGET"
rm -rf "$TARGET"/*
cp -R ./demo/* "$TARGET/"

echo "Publish complete to $TARGET"