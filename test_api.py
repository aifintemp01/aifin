#!/usr/bin/env python3
"""Test script to verify the agents API endpoint"""

import sys
import os
import requests

# Add the project root to the Python path
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)

def test_agents_api():
    try:
        response = requests.get("http://localhost:8000/hedge-fund/agents")
        response.raise_for_status()
        
        data = response.json()
        agents = data.get("agents", [])
        
        print(f"API returned {len(agents)} agents")
        print()
        
        print("Agent keys:")
        for agent in agents:
            print(f"- {agent['key']}")
        print()
        
        # Check if liquidity agent is present
        liquidity_agent = next((a for a in agents if a['key'] == 'liquidity'), None)
        if liquidity_agent:
            print("Liquidity agent found:")
            print(f"  Key: {liquidity_agent['key']}")
            print(f"  Display name: {liquidity_agent['display_name']}")
            print(f"  Description: {liquidity_agent['description']}")
            print(f"  Order: {liquidity_agent['order']}")
        else:
            print("Liquidity agent NOT found")
            
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        return False
        
    return True

if __name__ == "__main__":
    print("Testing /hedge-fund/agents API endpoint...")
    print("-" * 50)
    success = test_agents_api()
    print("-" * 50)
    if success:
        print("Test completed successfully")
    else:
        print("Test failed")
