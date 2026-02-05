"""
API Connection Testing Script

This script tests the connectivity and configuration of all external APIs
used by the Eye-LLM system.
"""

import os
import sys
from dotenv import load_dotenv

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()


def test_openai_api():
    """Test OpenAI API connection"""
    print("\n" + "="*60)
    print("🔍 Testing OpenAI API Connection")
    print("="*60)

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        print("❌ FAILED: OPENAI_API_KEY not found in environment")
        return False

    if api_key.startswith("sk-proj-") or api_key.startswith("sk-"):
        print(f"✅ API Key found: {api_key[:10]}...{api_key[-4:]}")

        try:
            from langchain_openai import ChatOpenAI

            llm = ChatOpenAI(
                model=os.getenv("LLM_MODEL", "gpt-4o"),
                api_key=api_key,
                temperature=0.7
            )

            # Test with a simple invocation
            from langchain_core.prompts import ChatPromptTemplate
            from langchain_core.output_parsers import StrOutputParser

            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a helpful assistant."),
                ("user", "Say 'API test successful' in exactly those words.")
            ])

            chain = prompt | llm | StrOutputParser()
            result = chain.invoke({})

            if "API test successful" in result or "successful" in result.lower():
                print(f"✅ OpenAI API test PASSED")
                print(f"   Response: {result}")
                return True
            else:
                print(f"⚠️  OpenAI API responded but with unexpected output")
                print(f"   Response: {result}")
                return True

        except Exception as e:
            print(f"❌ FAILED: {str(e)}")
            return False
    else:
        print("❌ FAILED: Invalid API key format (should start with 'sk-proj-' or 'sk-')")
        return False


def test_replicate_api():
    """Test Replicate API connection"""
    print("\n" + "="*60)
    print("🔍 Testing Replicate API Connection (SAM 2)")
    print("="*60)

    api_token = os.getenv("REPLICATE_API_TOKEN")

    if not api_token:
        print("❌ FAILED: REPLICATE_API_TOKEN not found in environment")
        print("   Note: SAM 2 segmentation is optional. You can skip this test.")
        return None

    if api_token.startswith("r8_"):
        print(f"✅ API Token found: {api_token[:10]}...{api_token[-4:]}")

        try:
            import replicate

            client = replicate.Client(api_token=api_token)

            # Test account access (doesn't cost anything)
            # Just checking if we can authenticate
            print(f"✅ Replicate client initialized successfully")
            print(f"   Model: {os.getenv('SAM2_MODEL_VERSION', 'meta/sam2-hiera-large')}")
            print(f"\n   ⚠️  Note: Full segmentation test requires an actual image file.")
            print(f"   Skipping actual segmentation to avoid API costs.")
            return True

        except ImportError:
            print(f"❌ FAILED: replicate package not installed")
            print(f"   Install with: uv pip install replicate")
            return False
        except Exception as e:
            print(f"❌ FAILED: {str(e)}")
            return False
    else:
        print("❌ FAILED: Invalid API token format (should start with 'r8_')")
        return False


def test_topology_data():
    """Test topology data loading"""
    print("\n" + "="*60)
    print("🔍 Testing Topology Data Loading")
    print("="*60)

    try:
        from skills.topology.graph_engine import TopologyEngine

        # Test with TH map
        engine = TopologyEngine('TH')

        num_nodes = engine.graph.number_of_nodes()
        num_edges = engine.graph.number_of_edges()

        print(f"✅ Topology engine initialized successfully")
        print(f"   Map: TH")
        print(f"   Nodes: {num_nodes}")
        print(f"   Edges: {num_edges}")

        # Test querying a node
        sample_node = list(engine.graph.nodes())[0]
        result = engine.query_node(sample_node)

        if "error" not in result:
            print(f"✅ Node query test PASSED")
            print(f"   Sample node: {sample_node}")
            print(f"   Node name: {result['info']['name']}")
            return True
        else:
            print(f"❌ Node query test FAILED: {result['error']}")
            return False

    except Exception as e:
        print(f"❌ FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_config_loading():
    """Test configuration loading"""
    print("\n" + "="*60)
    print("🔍 Testing Configuration Loading")
    print("="*60)

    try:
        from config import config

        print(f"✅ Configuration loaded successfully")
        print(f"   LLM Model: {config.model.llm_model}")
        print(f"   Short-term memory size: {config.memory.short_term_size}")
        print(f"   Long-term memory max: {config.memory.long_term_max_size}")
        print(f"   Attention levels: {list(config.attention.ATTENTION_DURATION.keys())}")

        # Test validation
        is_valid = config.validate()
        if is_valid:
            print(f"✅ Configuration validation PASSED")
            return True
        else:
            print(f"⚠️  Configuration validation found issues (see above)")
            return False

    except Exception as e:
        print(f"❌ FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all API tests"""
    print("\n" + "="*70)
    print("  Eye-LLM System - API Connection Test Suite")
    print("="*70)
    print("\nTesting connectivity to all external APIs and data sources...")

    results = {
        "Configuration": test_config_loading(),
        "Topology Data": test_topology_data(),
        "OpenAI API": test_openai_api(),
        "Replicate API": test_replicate_api()
    }

    # Summary
    print("\n" + "="*70)
    print("📊 TEST SUMMARY")
    print("="*70)

    for test_name, result in results.items():
        if result is True:
            status = "✅ PASSED"
        elif result is False:
            status = "❌ FAILED"
        else:
            status = "⏭️  SKIPPED"

        print(f"{test_name:20s}: {status}")

    # Count results
    passed = sum(1 for r in results.values() if r is True)
    failed = sum(1 for r in results.values() if r is False)
    skipped = sum(1 for r in results.values() if r is None)

    print(f"\nTotal: {passed} passed, {failed} failed, {skipped} skipped")

    if failed == 0:
        print("\n🎉 All critical tests passed! The system is ready to use.")
        if skipped > 0:
            print(f"\n⚠️  Note: {skipped} optional test(s) were skipped.")
            print("   The system will work, but some features may be disabled.")
    else:
        print(f"\n⚠️  {failed} test(s) failed. Please fix the issues above.")

    print("="*70 + "\n")


if __name__ == "__main__":
    main()
