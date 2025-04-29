# Checkers AI with Reinforcement Learning

This project implements an AI agent for playing Checkers, originally developed around 2018 and since migrated to PyTorch. It utilizes reinforcement learning techniques inspired by AlphaZero, combining a Residual Neural Network (ResNet) with Monte Carlo Tree Search (MCTS).

## Project Structure

*   `checkersTrainer.py`: The main script for training and evaluating the AI agent. Orchestrates self-play, model training, and matches against a benchmark opponent.
*   `TDAgent.py`: Implements the primary learning agent. It manages the neural network, coordinates self-play using MCTS, generates training data, and updates the network.
*   `mcts.py`: Contains the Monte Carlo Tree Search implementation. It uses the neural network for node evaluation and policy guidance (optional). Features include dynamic search termination based on KL-divergence and potential Transposition-Enhanced MCTS (TEMCTS) elements (alpha-beta bounds).
*   `resNN.py`: Defines the Residual Neural Network architecture using **PyTorch**. Takes the board state as input and outputs a predicted game value and (optionally) move probabilities. Uses `channels_first` data format (`(batch, channels, height, width)`).
*   `cnn.py`: An alternative, likely older, CNN architecture, also migrated to PyTorch.
*   `checkersBoard.py`: Implements the Checkers game logic, including board representation, move validation, jump handling, game rules, and win/loss/draw conditions.
*   `minimaxAgent.py`: A traditional minimax agent with alpha-beta pruning, used as a benchmark opponent during evaluation. Uses a heuristic evaluation function.
*   `QLearner.py`: An implementation of tabular Q-learning, potentially used for earlier experiments or baselines (currently commented out in `checkersTrainer.py`).
*   `utils.py`: Utility functions.
*   `createNNModelFile.py`: Script to create and save an initial **PyTorch** neural network model file (`.pth`).
*   `requirements.txt`: Lists the required Python packages.
*   `pytorch_migration_plan.md`: Describes the steps taken to migrate from TensorFlow/Keras to PyTorch.

## Core Concepts

*   **Self-Play:** The `TDAgent` learns by playing games against itself. MCTS is used to select moves during these games.
*   **MCTS:** The search algorithm explores possible game trajectories, guided by the neural network's predictions. Visit counts from the search determine the move probabilities for training data.
*   **Neural Network:** A ResNet implemented in **PyTorch** predicts the expected outcome (value) of a given board state and optionally a probability distribution over possible moves (policy).
*   **Training:** The network is trained on data generated from self-play games. The data consists of board states, the final game result (as the value target), and the MCTS move probabilities (as the policy target, if applicable).
*   **Evaluation:** The trained agent (`TDAgent`) plays against the `minimaxAgent` to assess its strength. The difficulty of the minimax opponent increases as the agent improves.

## How to Run

1.  **Dependencies:** Ensure you have Python installed. Create a virtual environment and install the required packages:
    ```bash
    python -m venv venv_pytorch
    source venv_pytorch/bin/activate  # On Windows use `venv_pytorch\Scripts\activate`
    pip install -r requirements.txt
    ```
    Make sure you have **uninstalled** TensorFlow and Keras if they were previously present in your environment to avoid conflicts.
2.  **Model File:** An initial model file (e.g., `res64x3.pth`) is required. Run the creation script if starting from scratch:
    ```bash
    python createNNModelFile.py --output res64x3.pth
    ```
3.  **Training/Evaluation:** Run the main trainer script:
    ```bash
    python checkersTrainer.py
    ```
4.  **Configuration:** Adjust parameters within `checkersTrainer.py` (e.g., `iterations`, `test_games`, `model_file`, `opponent_depth`, `q_learning_only`, `target_average_num_sims`).

## Notes

*   The codebase now uses **PyTorch**. TensorFlow/Keras specific code (like `tf.ConfigProto`, `keras.backend.set_session`, `.h5` model files) has been removed.
*   The MCTS implementation includes dynamic termination based on KL divergence, which adapts the search computation based on probability stabilization.
*   The project leverages multiprocessing in `TDAgent.py` to accelerate self-play game generation. The model architecture parameters are passed to worker processes during initialization.
*   Convolutional layers expect `channels_first` data format (`(batch, channels, height, width)`).