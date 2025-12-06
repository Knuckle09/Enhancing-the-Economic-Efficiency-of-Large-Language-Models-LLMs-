"""
Gemini API Setup Helper
This script helps you set up and test your Gemini API key

Usage:
    python setup_gemini.py --api-key YOUR_API_KEY
    python setup_gemini.py --test
"""

import os
import sys
import argparse


def set_api_key(api_key):
    """Set Gemini API key in environment"""
    os.environ['GEMINI_API_KEY'] = api_key
    print("✅ Gemini API key set for current session")
    print("\nTo make it permanent:")
    print("\nLinux/Mac:")
    print(f'  echo "export GEMINI_API_KEY=\'{api_key}\'" >> ~/.bashrc')
    print("  source ~/.bashrc")
    print("\nWindows PowerShell:")
    print(f'  [System.Environment]::SetEnvironmentVariable("GEMINI_API_KEY", "{api_key}", "User")')
    print("\nWindows Command Prompt:")
    print(f'  setx GEMINI_API_KEY "{api_key}"')


def test_gemini_connection():
    """Test Gemini API connection"""
    print("🔍 Testing Gemini API connection...")
    
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("❌ GEMINI_API_KEY environment variable not found")
        print("\nPlease set your API key first:")
        print("  python setup_gemini.py --api-key YOUR_API_KEY")
        return False
    
    try:
        import google.generativeai as genai
        print("✅ google-generativeai package is installed")
    except ImportError:
        print("❌ google-generativeai package not found")
        print("Install it with: pip install google-generativeai")
        return False
    
    try:
        genai.configure(api_key=api_key)
        print("✅ API key configured successfully")
        
        print("\n🧪 Testing with a simple prompt...")
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content("Say hello!")
        
        print("✅ Gemini API is working!")
        print(f"\nTest response: {response.text[:100]}...")
        return True
        
    except Exception as e:
        print(f"❌ Gemini API test failed: {e}")
        print("\nPossible issues:")
        print("  1. Invalid API key")
        print("  2. API key doesn't have proper permissions")
        print("  3. Network connectivity issues")
        print("  4. API quota exceeded")
        return False


def show_available_models():
    """Show available Gemini models"""
    print("\n📋 Available Gemini Models in this framework:")
    print("\n1. gemini-pro")
    print("   - Best for: Text generation, coding, math, reasoning")
    print("   - Alias: 'gemini'")
    print("\n2. gemini-1.5-flash")
    print("   - Best for: Fast responses, lighter tasks")
    print("   - Alias: 'gemini-flash'")
    print("\nUsage in code:")
    print('  llm_tester.generate_response("Your prompt", llm="gemini-pro")')
    print('  llm_tester.generate_response("Your prompt", llm="gemini-flash")')


def create_env_file(api_key):
    """Create a .env file with the API key"""
    env_content = f"""# Gemini API Configuration
GEMINI_API_KEY={api_key}

# Optional: Set preferred LLM for different tasks
# PREFERRED_CODING_LLM=gemini-pro
# PREFERRED_MATH_LLM=gemini-pro
# PREFERRED_GENERIC_LLM=gemini-flash
"""
    
    try:
        with open('.env', 'w') as f:
            f.write(env_content)
        print("✅ Created .env file with your API key")
        print("⚠️  Note: Add .env to your .gitignore to keep your API key private!")
        
        gitignore_path = '.gitignore'
        if os.path.exists(gitignore_path):
            with open(gitignore_path, 'r') as f:
                gitignore_content = f.read()
            if '.env' not in gitignore_content:
                with open(gitignore_path, 'a') as f:
                    f.write('\n# Environment variables\n.env\n')
                print("✅ Added .env to .gitignore")
        else:
            with open(gitignore_path, 'w') as f:
                f.write('# Environment variables\n.env\n')
            print("✅ Created .gitignore with .env")
            
    except Exception as e:
        print(f"❌ Failed to create .env file: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Setup and test Gemini API integration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Set API key for current session
  python setup_gemini.py --api-key YOUR_API_KEY
  
  # Create .env file with API key
  python setup_gemini.py --api-key YOUR_API_KEY --save-env
  
  # Test connection
  python setup_gemini.py --test
  
  # Show available models
  python setup_gemini.py --models
  
Get your API key from: https://makersuite.google.com/app/apikey
        """
    )
    
    parser.add_argument(
        '--api-key',
        type=str,
        help='Your Gemini API key'
    )
    
    parser.add_argument(
        '--save-env',
        action='store_true',
        help='Save API key to .env file'
    )
    
    parser.add_argument(
        '--test',
        action='store_true',
        help='Test Gemini API connection'
    )
    
    parser.add_argument(
        '--models',
        action='store_true',
        help='Show available Gemini models'
    )
    
    args = parser.parse_args()
    
    if len(sys.argv) == 1:
        parser.print_help()
        print("\n" + "="*60)
        print("Quick Start:")
        print("="*60)
        print("1. Get API key: https://makersuite.google.com/app/apikey")
        print("2. Set it: python setup_gemini.py --api-key YOUR_KEY --save-env")
        print("3. Test it: python setup_gemini.py --test")
        return
    
    if args.api_key:
        set_api_key(args.api_key)
        if args.save_env:
            create_env_file(args.api_key)
    
    if args.test:
        test_gemini_connection()
    
    if args.models:
        show_available_models()


if __name__ == "__main__":
    main()
