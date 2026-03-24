#!/usr/bin/env python3
"""Test script to verify the analysts configuration"""

import sys
import os

# Add the project root to the Python path
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)

from src.utils.analysts import ANALYST_CONFIG, get_agents_list

print("=== ANALYST_CONFIG ===")
print(f"Number of analysts: {len(ANALYST_CONFIG)}")
print()

print("=== Analyst Keys ===")
for key in sorted(ANALYST_CONFIG.keys()):
    print(f"- {key}")
print()

print("=== get_agents_list() ===")
agents_list = get_agents_list()
print(f"Number of agents: {len(agents_list)}")
print()

for agent in agents_list:
    print(f"Key: {agent['key']}")
    print(f"Name: {agent['display_name']}")
    print(f"Description: {agent['description']}")
    print(f"Order: {agent['order']}")
    print()

# Check if liquidity agent is present
print("=== Liquidity Agent Check ===")
liquidity_agent = next((a for a in agents_list if a['key'] == 'liquidity'), None)
if liquidity_agent:
    print("OK: Liquidity agent found")
    print(f"  - Display name: {liquidity_agent['display_name']}")
    print(f"  - Description: {liquidity_agent['description']}")
else:
    print("ERROR: Liquidity agent not found")
    print()
    print("Available agents:")
    for agent in agents_list:
        print(f"  - {agent['key']}: {agent['display_name']}")
