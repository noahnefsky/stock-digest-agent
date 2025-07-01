#!/usr/bin/env python3
"""
Test script for the React agent implementation
"""

import os
from dotenv import load_dotenv
from agent import StockDigestAgent

def test_react_agent():
    """Test the React agent setup"""
    load_dotenv()
    
    print("Testing React agent setup...")
    
    # Create agent instance
    agent = StockDigestAgent()
    
    print("✓ Agent created successfully")
    print(f"✓ React tools: {len(agent.react_tools)} tools configured")
    print(f"✓ React agent: {type(agent.react_agent).__name__}")
    print(f"✓ React executor: {type(agent.react_agent_executor).__name__}")
    
    # Test a simple search
    print("\nTesting simple search...")
    try:
        result = agent.react_agent_executor.invoke({"input": "Find recent news about AAPL stock"})
        print(f"✓ Search completed: {type(result)}")
        print(f"✓ Output type: {type(result.get('output', ''))}")
        print(f"✓ Output length: {len(str(result.get('output', '')))} characters")
    except Exception as e:
        print(f"✗ Search failed: {e}")
    
    print("\nReact agent test completed!")

if __name__ == "__main__":
    test_react_agent() 