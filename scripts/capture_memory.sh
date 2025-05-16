#!/bin/bash
# capture_memory.sh - Script to capture memory usage during inference
#
# This script captures screenshots at regular intervals and converts them
# to a GIF, allowing you to visualize memory reduction in Activity Monitor
# or other monitoring tools when running different KV cache configurations.
#
# Usage:
#   ./scripts/capture_memory.sh [options]
#
# Options:
#   --frames N    Number of frames to capture (default: 30)
#   --delay N     Delay between frames in seconds (default: 1)
#   --fps N       Frames per second in the output GIF (default: 5)
#   --output FILE Output filename (default: memory_reduction.gif)

set -euo pipefail

# ANSI color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
RESET='\033[0m'

# Default settings
FRAMES=30
DELAY=1
FPS=5
OUTPUT="memory_reduction.gif"
FRAMES_DIR="capture_frames"

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --frames)
      FRAMES="$2"
      shift 2
      ;;
    --delay)
      DELAY="$2"
      shift 2
      ;;
    --fps)
      FPS="$2"
      shift 2
      ;;
    --output)
      OUTPUT="$2"
      shift 2
      ;;
    *)
      echo -e "${RED}Error: Unknown option $1${RESET}"
      exit 1
      ;;
  esac
done

# Check for gifski
if ! command -v gifski &> /dev/null; then
    echo -e "${YELLOW}⚠️  gifski is not installed. Attempting to install it...${RESET}"
    if command -v brew &> /dev/null; then
        brew install gifski
    else
        echo -e "${RED}❌ Error: Homebrew is not installed. Please install gifski manually:${RESET}"
        echo -e "${BLUE}   brew install gifski${RESET}"
        exit 1
    fi
fi

echo -e "${GREEN}📹 Memory Usage Capture Tool${RESET}"
echo -e "${BLUE}This script will capture ${FRAMES} screenshots at ${DELAY}-second intervals${RESET}"
echo -e "${BLUE}and combine them into a GIF at ${FPS} frames per second.${RESET}"
echo
echo -e "${YELLOW}Instructions:${RESET}"
echo -e "1. Open Activity Monitor (Applications > Utilities > Activity Monitor)"
echo -e "2. Position it on your screen where it's clearly visible"
echo -e "3. Sort by Memory usage (click the 'Memory' column header)"
echo -e "4. Run your KVSplit commands in another terminal window"
echo

# Ask for confirmation
read -p "Press Enter when ready to start capturing, or Ctrl+C to cancel..."

# Create frames directory
mkdir -p "$FRAMES_DIR"

# Start countdown
echo -e "${YELLOW}Starting capture in:${RESET}"
for i in {3..1}; do
    echo -e "${YELLOW}$i...${RESET}"
    sleep 1
done

echo -e "${GREEN}🎬 Capturing started!${RESET}"

# Capture frames
for ((i=1; i<=$FRAMES; i++)); do
    echo -e "${BLUE}Capturing frame $i of $FRAMES${RESET}"
    screencapture -x "$FRAMES_DIR/frame_$(printf "%03d" $i).png"
    
    # Show progress
    percent=$((i * 100 / FRAMES))
    completed=$((percent / 2))
    remaining=$((50 - completed))
    progress="["
    for ((j=0; j<completed; j++)); do progress+="="; done
    progress+=">"
    for ((j=0; j<remaining; j++)); do progress+=" "; done
    progress+="] $percent%"
    echo -e "${GREEN}$progress${RESET}"
    
    sleep "$DELAY"
done

echo -e "${GREEN}✅ Capture complete!${RESET}"
echo -e "${BLUE}Converting frames to GIF...${RESET}"

# Convert frames to GIF
if ! gifski --fps "$FPS" "$FRAMES_DIR"/frame_*.png -o "$OUTPUT"; then
    echo -e "${RED}❌ Error: Failed to create GIF. Check if gifski is installed correctly.${RESET}"
    echo -e "${YELLOW}⚠️  Frame images are saved in $FRAMES_DIR directory.${RESET}"
    exit 1
fi

echo -e "${GREEN}✅ GIF saved as $OUTPUT${RESET}"
echo -e "${BLUE}Frame images are saved in $FRAMES_DIR directory.${RESET}"
echo

# Ask if user wants to clean up frame images
read -p "Do you want to remove the individual frame images? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    rm -rf "$FRAMES_DIR"
    echo -e "${GREEN}Frame directory removed.${RESET}"
else
    echo -e "${BLUE}Frame directory kept at $FRAMES_DIR${RESET}"
fi

echo
echo -e "${YELLOW}Tip: To create a more compelling visualization:${RESET}"
echo -e "1. Run FP16 (baseline) configuration first"
echo -e "2. While it's running, start capturing frames"
echo -e "3. Switch to K8V4 or K4V4 configuration mid-capture"
echo -e "4. The resulting GIF will show the memory drop when switching configurations${RESET}"
