#!/usr/bin/env python3
"""
Test script for MCP integration with Tavily
"""

import os
import asyncio
from dotenv import load_dotenv
from mcp_use import MCPAgent, MCPClient
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

def test_mcp_integration():
    """Test the MCP integration with Tavily"""
    
    # Initialize the LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.1,
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    
    # Initialize MCP client for Tavily
    mcp_config = {
        "mcpServers": {
            "tavily-mcp": {
                "command": "npx",
                "args": ["-y", "tavily-mcp@0.1.3"],
                "env": {
                    "TAVILY_API_KEY": os.getenv("TAVILY_API_KEY")
                }
            }
        }
    }
    
    try:
        # Create MCP client
        client = MCPClient.from_dict(mcp_config)
        
        # Create agent with the client
        agent = MCPAgent(llm=llm, client=client, max_steps=30)
        
        # Test search query
        test_query = "Search for Apple stock news"
        print(f"Testing MCP search with query: {test_query}")
        
        result = agent.run(test_query)
        print("MCP search result:")
        print(result)
        
        return True
        
    except Exception as e:
        print(f"Error testing MCP integration: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_mcp_integration()
    if success:
        print("MCP integration test passed!")
    else:
        print("MCP integration test failed!") 