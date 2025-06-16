#!/bin/bash
#
# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/

# Helper script to install HMMER (http://hmmer.org/) for AlphaFold 3.
# This script attempts to install HMMER using common package managers
# or provides instructions for manual installation.

set -e

echo "Attempting to install HMMER..."
echo "Please ensure you have sudo privileges if required by your system's package manager."

# --- macOS ---
if [[ "$(uname)" == "Darwin" ]]; then
    echo "Detected macOS."
    if command -v brew &> /dev/null; then
        echo "Homebrew detected. Attempting to install HMMER using Homebrew..."
        brew install hmmer
        echo "HMMER installation via Homebrew attempted."
        echo "Please verify the installation by running: jackhmmer -h"
        exit 0
    else
        echo "Homebrew not found. Please install Homebrew first (see https://brew.sh/)"
        echo "or install HMMER manually from http://hmmer.org/download.html"
        exit 1
    fi
fi

# --- Linux ---
if [[ "$(uname)" == "Linux" ]]; then
    echo "Detected Linux."
    # Try apt-get (Debian/Ubuntu)
    if command -v apt-get &> /dev/null; then
        echo "Attempting to install HMMER using apt-get..."
        sudo apt-get update
        sudo apt-get install -y hmmer
        echo "HMMER installation via apt-get attempted."
        echo "Please verify the installation by running: jackhmmer -h"
        exit 0
    # Try yum (CentOS/RHEL)
    elif command -v yum &> /dev/null; then
        echo "Attempting to install HMMER using yum..."
        # HMMER might be in EPEL repository for older CentOS/RHEL
        # sudo yum install -y epel-release # Uncomment if HMMER is not found
        sudo yum install -y hmmer
        echo "HMMER installation via yum attempted."
        echo "Please verify the installation by running: jackhmmer -h"
        exit 0
    else
        echo "Could not find apt-get or yum."
    fi
fi

# --- Fallback/Manual Installation ---
echo ""
echo "Could not automatically install HMMER using common package managers for your system."
echo "Please install HMMER manually:"
echo "1. Visit the HMMER download page: http://hmmer.org/download.html"
echo "2. Download the latest stable release (HMMER3, version 3.1b2 or later)."
echo "3. Follow the installation instructions provided in the HMMER package."
echo "   This typically involves commands like:"
echo "   ./configure"
echo "   make"
echo "   sudo make install"
echo "4. Ensure the HMMER binaries (jackhmmer, nhmmer, etc.) are in your system's PATH."
echo ""
echo "Alternatively, if you use Conda:"
echo "conda install -c bioconda hmmer"
echo ""
echo "After installation, verify by running: jackhmmer -h"

exit 1
