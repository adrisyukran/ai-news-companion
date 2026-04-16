"""
AI News Companion - Manual Connectivity Test

Run this test manually after setting NANO_GPT_API_KEY environment variable:

    set NANO_GPT_API_KEY=your_api_key_here
    python tests/test_llm_connectivity.py

Or use a .env file with python-dotenv.
"""
import asyncio
import os
import sys

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.services.llm_service import NanoGPTService


async def test_connectivity():
    """Test basic API connectivity."""
    print("=" * 60)
    print("AI News Companion - nano-gpt Connectivity Test")
    print("=" * 60)
    
    # Check API key
    api_key = os.environ.get("NANO_GPT_API_KEY", "")
    if not api_key:
        print("\n[ERROR] NANO_GPT_API_KEY environment variable is not set!")
        print("\nTo set it:")
        print("  Windows: set NANO_GPT_API_KEY=your_api_key_here")
        print("  Linux/Mac: export NANO_GPT_API_KEY=your_api_key_here")
        return False
    
    print(f"\n[INFO] API Key found: {api_key[:8]}...{api_key[-4:]}")
    
    # Create service
    service = NanoGPTService()
    
    try:
        # Test 1: Basic connectivity
        print("\n[Test 1] Testing basic connectivity...")
        result = await service.test_connectivity()
        if result:
            print("[PASS] Connectivity test successful!")
        else:
            print("[FAIL] Connectivity test failed!")
            return False
        
        # Test 2: Simple completion
        print("\n[Test 2] Testing simple completion...")
        response = await service.complete(
            prompt="What is 2 + 2? Answer with only the number.",
            max_tokens=10,
        )
        print(f"[INFO] Response: {response.content}")
        print(f"[INFO] Model: {response.model}")
        print("[PASS] Simple completion successful!")
        
        # Test 3: System prompt
        print("\n[Test 3] Testing with system prompt...")
        response = await service.complete(
            prompt="Translate 'Hello' to Bahasa Melayu.",
            system_prompt="You are a helpful translation assistant.",
            max_tokens=20,
        )
        print(f"[INFO] Response: {response.content}")
        print("[PASS] System prompt test successful!")
        
        # Test 4: Streaming
        print("\n[Test 4] Testing streaming completion...")
        chunks = []
        async for chunk in service.complete_stream(
            prompt="Count from 1 to 3, one number per line.",
            max_tokens=20,
        ):
            chunks.append(chunk)
            print(f"[STREAM] Received chunk: {chunk}", end="", flush=True)
        
        print()
        print(f"[INFO] Total chunks received: {len(chunks)}")
        print("[PASS] Streaming test successful!")
        
        # Test 5: Long text chunking
        print("\n[Test 5] Testing chunking with long text...")
        long_text = "This is a test article. " * 500  # ~5000 tokens
        chunks = service._chunk_text(long_text)
        print(f"[INFO] Split into {len(chunks)} chunks")
        for i, chunk in enumerate(chunks[:3]):  # Show first 3 chunks
            print(f"[INFO] Chunk {i+1}: {len(chunk)} chars")
        print("[PASS] Chunking test successful!")
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"\n[ERROR] Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        await service.close()


if __name__ == "__main__":
    success = asyncio.run(test_connectivity())
    sys.exit(0 if success else 1)
