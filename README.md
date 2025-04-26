# Checkers AI with Reinforcement Learning

This project implements an AI agent for playing Checkers, developed around 2018. It utilizes reinforcement learning techniques inspired by AlphaZero, combining a Residual Neural Network (ResNet) with Monte Carlo Tree Search (MCTS).

## Project Structure

*   `checkersTrainer.py`: The main script for training and evaluating the AI agent. Orchestrates self-play, model training, and matches against a benchmark opponent.
*   `TDAgent.py`: Implements the primary learning agent. It manages the neural network, coordinates self-play using MCTS, generates training data, and updates the network.
*   `mcts.py`: Contains the Monte Carlo Tree Search implementation. It uses the neural network for node evaluation and policy guidance (optional). Features include dynamic search termination based on KL-divergence and potential Transposition-Enhanced MCTS (TEMCTS) elements (alpha-beta bounds).
*   `resNN.py`: Defines the Residual Neural Network architecture using Keras/TensorFlow. Takes the board state as input and outputs a predicted game value and (optionally) move probabilities.
*   `cnn.py`: An alternative, likely older, CNN architecture.
*   `checkersBoard.py`: Implements the Checkers game logic, including board representation, move validation, jump handling, game rules, and win/loss/draw conditions.
*   `minimaxAgent.py`: A traditional minimax agent with alpha-beta pruning, used as a benchmark opponent during evaluation. Uses a heuristic evaluation function.
*   `QLearner.py`: An implementation of tabular Q-learning, potentially used for earlier experiments or baselines (currently commented out in `checkersTrainer.py`).
*   `utils.py`: Utility functions.
*   `createNNModelFile.py`: Script to create and save an initial neural network model file (`.h5`).

## Core Concepts

*   **Self-Play:** The `TDAgent` learns by playing games against itself. MCTS is used to select moves during these games.
*   **MCTS:** The search algorithm explores possible game trajectories, guided by the neural network's predictions. Visit counts from the search determine the move probabilities for training data.
*   **Neural Network:** A ResNet predicts the expected outcome (value) of a given board state and optionally a probability distribution over possible moves (policy).
*   **Training:** The network is trained on data generated from self-play games. The data consists of board states, the final game result (as the value target), and the MCTS move probabilities (as the policy target, if applicable).
*   **Evaluation:** The trained agent (`TDAgent`) plays against the `minimaxAgent` to assess its strength. The difficulty of the minimax opponent increases as the agent improves.

## How to Run (Potential Steps)

1.  **Dependencies:** Ensure you have Python installed, along with libraries like TensorFlow (likely TF 1.x given the age), Keras, NumPy, and Matplotlib. A `requirements.txt` might be needed.
2.  **Model File:** A pre-trained or initial model file (e.g., `res128x5.h5` mentioned in `checkersTrainer.py`) is required. You might need to run `createNNModelFile.py` if starting from scratch.
3.  **Training/Evaluation:** Run the main trainer script:
    ```bash
    python checkersTrainer.py
    ```
4.  **Configuration:** Adjust parameters within `checkersTrainer.py` (e.g., `iterations`, `test_games`, `model_file`, `opponent_depth`, `q_learning_only`).

## Notes

*   The code uses `tensorflow.ConfigProto()` and `keras.backend.set_session()`, indicating it was written for TensorFlow 1.x. Compatibility with TensorFlow 2.x might require modifications.
*   The MCTS implementation includes dynamic termination based on KL divergence, which adapts the search computation based on probability stabilization.
*   The project leverages multiprocessing in `TDAgent.py` to accelerate self-play game generation.