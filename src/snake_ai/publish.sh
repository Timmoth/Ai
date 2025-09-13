#!/bin/bash
set -e


TARGET=../../demos/snake_ai

mkdir -p "$TARGET"
rm -rf "$TARGET"/*
cp -R ./demo/* "$TARGET/"

echo "Publish complete to $TARGET"