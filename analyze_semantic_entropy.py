from src.semantic_entropy_agent import SemanticEntropyAgent
import argparse

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Analyze semantic entropy of conversations')
    parser.add_argument('--start', type=int, default=0, help='Start conversation ID')
    parser.add_argument('--end', type=int, default=None, help='End conversation ID')
    parser.add_argument('--test-file', type=str, help='Path to a test conversation JSON file')
    args = parser.parse_args()

    # Create semantic entropy agent
    agent = SemanticEntropyAgent()
    
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
        # Display the plots
        results['local_plot'].show()  # This will open the plot in your default browser
        results['global_plot'].show()

        # Print the average semantic entropy statistics
        print("\nLocal Semantic Entropy Statistics:")
        for speaker, stats in results['average_local'].items():
            print(f"\n{speaker}:")
            print(f"  Mean: {stats['mean']}")
            print(f"  Standard Deviation: {stats['std']}")
            
        print("\nGlobal Semantic Entropy Statistics:")
        for speaker, stats in results['average_global'].items():
            print(f"\n{speaker}:")
            print(f"  Mean: {stats['mean']}")
            print(f"  Standard Deviation: {stats['std']}")

if __name__ == "__main__":
    main() 