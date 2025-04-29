import torch
import resNN
import os
import argparse

def create_model(width=64, residual_blocks=3, q_learning_only=True, output_file='res64x3.pth'):
    """
    Create and save an initial PyTorch ResNN model.
    
    Args:
        width: Width of the neural network (number of filters in convolutional layers)
        residual_blocks: Number of residual blocks in the network
        q_learning_only: Whether to use only Q-learning (value head) or also policy head
        output_file: Path to save the model state dictionary
    """
    print(f"Creating ResNN model with width={width}, residual_blocks={residual_blocks}, q_learning_only={q_learning_only}")
    
    # Initialize the model
    model = resNN.ResNN(width=width, residual_blocks=residual_blocks, q_learning_only=q_learning_only)
    
    # Make sure the directory exists
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory {output_dir}")
    
    # Save the model state dictionary
    torch.save(model.state_dict(), output_file)
    print(f"Model saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create a new PyTorch ResNN model file.')
    parser.add_argument('--width', type=int, default=64, help='Width of the neural network (number of filters)')
    parser.add_argument('--blocks', type=int, default=3, help='Number of residual blocks')
    parser.add_argument('--q-learning-only', type=bool, default=True, help='Whether to use only value head (True) or also policy head (False)')
    parser.add_argument('--output', type=str, default='res64x3.pth', help='Output file path')
    
    args = parser.parse_args()
    create_model(args.width, args.blocks, args.q_learning_only, args.output)
