#!/bin/sh

set -e

npm ci

pipx install lit==18.1.8
pipx install filecheck==0.0.24
