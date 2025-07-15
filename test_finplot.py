#!/usr/bin/env python3
"""Test if finplot creates windows automatically."""

import finplot as fplt
import time

print("Testing finplot window creation...")

# Test 1: Just importing finplot
print("1. Imported finplot - any windows?")
time.sleep(2)

# Test 2: Create plot without show
print("2. Creating plot without show...")
ax = fplt.create_plot("Test Plot")
print("   Plot created - waiting 2 seconds...")
time.sleep(2)

# Test 3: Show the plot
print("3. Calling fplt.show()...")
fplt.show()
print("   Done")