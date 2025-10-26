#!/usr/bin/env python3
"""
GPT skin classification runner that works with metadata.csv
Provides the same workflow experience as Claude but uses your existing dataset.
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def main():
    """Run GPT classification on the metadata dataset."""
    
    # Check if API key is provided
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("Error: Please set the OPENAI_API_KEY environment variable")
        print("You can get your API key from: https://platform.openai.com/api-keys")
        print("\nExample:")
        print("Create a .env file with:")
        print("OPENAI_API_KEY=your-api-key-here")
        return
    
    # Check if metadata file exists
    metadata_file = "data/metadata.csv"
    if not os.path.exists(metadata_file):
        print(f"Error: Metadata file not found at {metadata_file}")
        print("Please run the ingest.py script first to create the metadata.")
        return
    
    print("🔬 GPT Skin Classification Tool")
    print("=" * 50)
    print(f"Metadata file: {metadata_file}")
    print("Output files:")
    print("  - results/predictions.csv (simple format)")
    print("  - results/gpt_detailed_results.json (detailed format)")
    
    # Ask user for confirmation
    print("\nThis will process skin condition images using OpenAI GPT-4o-mini.")
    print("The script will save intermediate results and handle rate limits.")
    print("⚠️  Note: This will use your OpenAI API quota and incur costs.")
    
    # For testing, you can limit the number of images
    test_mode = input("\nDo you want to run in test mode (process only 20 images)? (y/n): ").lower().strip()
    
    if test_mode == 'y':
        print("Running in test mode: processing 20 images")
        estimated_cost = 20 * 0.002  # Rough estimate
        print(f"Estimated cost: ~${estimated_cost:.3f}")
    else:
        print("Running in evaluation mode")
        print("You can specify the number of images when the script starts.")
        estimated_cost = 100 * 0.002  # Rough estimate for 100 images
        print(f"Estimated cost for 100 images: ~${estimated_cost:.2f}")
    
    confirm = input("Continue? (y/n): ").lower().strip()
    if confirm != 'y':
        print("Classification cancelled.")
        return
    
    # Import and run the main classification script
    try:
        print("\n" + "=" * 50)
        print("Starting GPT Classification...")
        print("=" * 50)
        
        # Import the main script
        sys.path.append('src')
        import run_llm
        
        # Run the main function
        run_llm.main()
        
        print("\n" + "=" * 50)
        print("✅ Classification complete!")
        print("=" * 50)
        print("\nNext steps:")
        print("1. Analyze results:")
        print("   python src/analyze_gpt_results.py --csv results/predictions.csv")
        print("\n2. Compare with Claude results (if available):")
        print("   python src/analyze_claude_results.py")
        print("\n3. Run more images:")
        print("   python src/run_gpt_metadata.py")
        
    except KeyboardInterrupt:
        print("\n⚠️  Classification interrupted by user")
        print("Partial results may be saved in results/ directory")
        
    except Exception as e:
        print(f"\n❌ Error during classification: {e}")
        print("Check your API key and internet connection.")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
