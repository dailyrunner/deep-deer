"""
Test script for Qwen model integration with LangChain
"""
import asyncio
import os
from app.services.llm_provider import LLMService
from app.services.langchain_pipeline import LangChainPipeline
from app.core.config import settings


async def test_qwen_provider():
    """Test Qwen provider with HuggingFacePipeline"""
    print("\n" + "="*60)
    print("Testing Qwen Provider with LangChain Integration")
    print("="*60)

    # Create LLM service
    llm_service = LLMService()

    # Test 1: Initialize with Qwen
    print("\n1. Initializing Qwen provider...")
    config = {
        "enable_qwen_local": True,
        "qwen_model": "Qwen/Qwen3-0.6B",
        "enable_ollama": False  # Disable Ollama to test Qwen only
    }

    success = await llm_service.initialize(config)
    print(f"   Initialization: {'✓ Success' if success else '✗ Failed'}")
    print(f"   Available providers: {llm_service.list_providers()}")
    print(f"   Default provider: {llm_service.get_default_provider()}")

    if not success:
        print("   Failed to initialize Qwen provider")
        return

    # Test 2: Direct generation
    print("\n2. Testing direct generation...")
    try:
        prompt = "What is artificial intelligence in one sentence?"
        response = await llm_service.generate(
            prompt=prompt,
            temperature=0.7,
            max_tokens=50
        )
        print(f"   Prompt: {prompt}")
        print(f"   Response: {response[:200]}...")
    except Exception as e:
        print(f"   ✗ Error: {e}")

    # Test 3: LangChain compatibility
    print("\n3. Testing LangChain compatibility...")
    try:
        provider = llm_service.providers.get("qwen_local")
        if provider:
            # Ensure LangChain LLM is created
            if hasattr(provider, '_ensure_langchain_llm'):
                llm = provider._ensure_langchain_llm()
                print(f"   LangChain LLM type: {type(llm)}")
                print(f"   Has invoke method: {hasattr(llm, 'invoke')}")

                # Test invoke
                test_response = llm.invoke("Hello, how are you?")
                print(f"   Test invoke response: {test_response[:100]}...")
            else:
                print("   ✗ No _ensure_langchain_llm method found")
    except Exception as e:
        print(f"   ✗ Error testing LangChain compatibility: {e}")

    # Test 4: Chain functionality
    print("\n4. Testing LangChain chain...")
    try:
        template = "Translate the following to {language}: {text}"
        response = await llm_service.generate_with_chain(
            prompt_template=template,
            input_variables={
                "language": "French",
                "text": "Hello world"
            }
        )
        print(f"   Chain response: {response}")
    except Exception as e:
        print(f"   ✗ Error in chain: {e}")

    await llm_service.cleanup()


async def test_pipeline_integration():
    """Test integration with LangChain pipeline"""
    print("\n" + "="*60)
    print("Testing LangChain Pipeline Integration")
    print("="*60)

    # Initialize LLM service with Qwen
    llm_service = LLMService()
    config = {
        "enable_qwen_local": True,
        "qwen_model": "Qwen/Qwen3-0.6B",
        "enable_ollama": False
    }

    await llm_service.initialize(config)

    # Initialize pipeline (should use Qwen from llm_service)
    pipeline = LangChainPipeline()
    await pipeline.initialize()

    print(f"\n✓ Pipeline initialized")
    print(f"  LLM type: {type(pipeline.llm)}")
    print(f"  Pipeline ready: {pipeline.is_ready}")

    if pipeline.is_ready:
        # Test a simple question
        try:
            print("\nTesting pipeline question answering...")
            answer = await pipeline.answer_question(
                "What is Python?",
                store_names=None
            )
            print(f"  Answer: {answer[:200]}...")
        except Exception as e:
            print(f"  ✗ Error: {e}")

    await llm_service.cleanup()


async def test_fallback_to_ollama():
    """Test fallback to Ollama when Qwen is not enabled"""
    print("\n" + "="*60)
    print("Testing Fallback to Ollama")
    print("="*60)

    # Initialize without Qwen
    llm_service = LLMService()
    config = {
        "enable_qwen_local": False,
        "enable_ollama": True,
        "ollama_model": "gpt-oss:120b"
    }

    await llm_service.initialize(config)
    print(f"\nProviders: {llm_service.list_providers()}")
    print(f"Default: {llm_service.get_default_provider()}")

    # Pipeline should use Ollama
    pipeline = LangChainPipeline()
    await pipeline.initialize()

    print(f"Pipeline LLM type: {type(pipeline.llm)}")

    await llm_service.cleanup()


async def main():
    """Run all tests"""
    print("\n" + "#"*60)
    print("# Qwen Model Integration Test Suite")
    print("#"*60)

    # Set environment for testing
    os.environ["ENABLE_QWEN_LOCAL"] = "true"
    os.environ["QWEN_MODEL"] = "Qwen/Qwen3-0.6B"

    try:
        # Test 1: Qwen provider
        await test_qwen_provider()

        # Test 2: Pipeline integration
        await test_pipeline_integration()

        # Test 3: Fallback mechanism
        await test_fallback_to_ollama()

        print("\n" + "#"*60)
        print("# All tests completed!")
        print("#"*60)

    except Exception as e:
        print(f"\n✗ Test suite failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())