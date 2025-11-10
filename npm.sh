#!/usr/bin/env bash
if [[ "$1" != "publish" ]]; then
  exit 0
fi
cp docs/README.md README.md
npm publish
rm README.md
