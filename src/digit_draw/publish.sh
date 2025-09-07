#!/bin/bash
set -e


TARGET=../../demos/digit_draw

mkdir -p "$TARGET"
rm -rf "$TARGET"/*
cp -R ./demo/* "$TARGET/"

echo "Publish complete to $TARGET"