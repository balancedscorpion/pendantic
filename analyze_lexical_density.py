from src.lexical_density_agent import LexicalDensityAgent
from src.bulk_lexical_density_agent import BulkLexicalDensityAgent
import argparse

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Analyze lexical density of conversations')
    parser.add_argument('--mode', choices=['bulk', 'normalized'], default='normalized',
                       help='Analysis mode: bulk (treat all text as one) or normalized (per speaker)')
    parser.add_argument('--start', type=int, default=0, help='Start conversation ID')
    parser.add_argument('--end', type=int, default=None, help='End conversation ID')
    parser.add_argument('--bin-size', type=int, default=5,
                       help='Number of utterances to combine in each bin (only used in bulk mode)')
    parser.add_argument('--test-file', type=str, help='Path to a test conversation JSON file')
    args = parser.parse_args()

    # Create appropriate agent based on mode
    if args.mode == 'bulk':
        agent = BulkLexicalDensityAgent()
        if args.test_file:
            results = agent.analyze_json_file_bulk(args.test_file, args.start, args.end, args.bin_size)
        else:
            # If no end_id specified, use 1000 as default for database queries
            end_id = args.end if args.end is not None else 1000
            results = agent.analyze_conversation_range_bulk(args.start, end_id, args.bin_size)
    else:
        agent = LexicalDensityAgent()
        if args.test_file:
            results = agent.analyze_json_file(args.test_file, args.start, args.end)
        else:
            # If no end_id specified, use 1000 as default for database queries
            end_id = args.end if args.end is not None else 1000
            results = agent.analyze_conversation_range(args.start, end_id)

    # Check if there was an error
    if 'error' in results:
        print(results['error'])
    else:
        # Display the plot
        results['plot'].show()  # This will open the plot in your default browser

        # Print the average lexical density statistics
        print(f"\nAverage Lexical Density Statistics ({args.mode} mode):")
        for speaker, stats in results['average_density'].items():
            print(f"\n{speaker}:")
            print(f"  Mean: {stats['mean']}%")
            print(f"  Standard Deviation: {stats['std']}%")

if __name__ == "__main__":
    main() 