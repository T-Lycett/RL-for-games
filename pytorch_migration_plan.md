# PyTorch Migration Plan (Full Rewrite, Channels First)

**Goal:** Replace TensorFlow/Keras with PyTorch, using `channels_first` data format for convolutional layers, and update all relevant parts of the codebase.

**Phase 1: Setup and Core NN Implementation**

1.  **Environment Setup:**
    *   Create a new Python virtual environment or update your existing one.
    *   Install PyTorch (`pip install torch torchvision torchaudio`).
    *   Ensure NumPy is installed (`pip install numpy`). Keep Matplotlib (`pip install matplotlib`).
    *   **Uninstall TensorFlow and Keras** (`pip uninstall tensorflow keras tensorflow-gpu` or similar, depending on how they were installed) to avoid conflicts and ensure you catch all dependencies.
    *   Create/update a `requirements.txt` file reflecting these changes.

2.  **Reimplement `ResNN` in PyTorch (`resNN.py`):**
    *   Create a new `resNN.py` or modify the existing one.
    *   Define a class `ResNN` inheriting from `torch.nn.Module`.
    *   In the `__init__` method:
        *   Define layers using `torch.nn`:
            *   `keras.layers.Conv2D` -> `torch.nn.Conv2d`. Specify `in_channels`, `out_channels`, `kernel_size`, `padding='same'`, `bias=False`. **PyTorch defaults to `channels_first`**, so the input tensor shape expected will be `(batch, channels, height, width)`.
            *   `keras.layers.BatchNormalization(axis=-1)` -> `torch.nn.BatchNorm2d(num_features=...)`. `BatchNorm2d` expects `channels_first` input, and `num_features` should be the number of channels.
            *   `keras.layers.ReLU` -> `torch.nn.ReLU()`.
            *   `keras.layers.Dense` -> `torch.nn.Linear(in_features=..., out_features=...)`.
            *   `keras.layers.Flatten` -> `torch.nn.Flatten()` or use `tensor.view(batch_size, -1)` in the `forward` method.
            *   `keras.layers.add` -> Direct tensor addition (`+`) in the `forward` method.
        *   Store the `width`, `residual_blocks`, and `q_learning_only` parameters.
    *   Implement the `residual_block` method using PyTorch layers, ensuring tensors are added correctly.
    *   Implement the `forward(self, x)` method:
        *   Define the data flow through the layers defined in `__init__`.
        *   Ensure the tensor shape is correctly handled, especially before and after `Flatten`/`Linear` layers (Conv layers output `(batch, channels, height, width)`, Linear expects `(batch, features)`).
        *   Return the `value` tensor, or both `value` and `probabilities` tensors if `q_learning_only` is `False`.

3.  **Adapt Feature Extraction (`TDAgent.py`):**
    *   Modify the static `extract_features` function in `TDAgent.py`.
    *   Currently, it uses `np.stack(..., axis=-1)` which creates a shape `(height, width, channels)`.
    *   Change the stacking to `np.stack(..., axis=0)` to produce the `channels_first` shape `(channels, height, width)` required by the PyTorch `Conv2d` layers. Update any comments mentioning the shape.

4.  **Update Model Creation Script (`createNNModelFile.py`):**
    *   Modify this script to instantiate the new PyTorch `ResNN`.
    *   Instead of `model.save(filename)`, save the initial model state dictionary: `torch.save(model.state_dict(), filename)`. Make sure the filename has a suitable extension like `.pt` or `.pth` (e.g., `res64x3.pth`).
    *   Update the `model_file` variable in `checkersTrainer.py` to use the new filename/extension.

**Phase 2: Training and Integration**

5.  **Rewrite Training Logic (`TDAgent.update_model`):**
    *   Remove the Keras `model.compile` calls (in `__init__` and `set_lr`). Learning rate is handled by the optimizer instance.
    *   Replace the `self.NN.fit(...)` call.
    *   Inside `update_model`:
        *   Instantiate a PyTorch optimizer (e.g., `optimizer = torch.optim.Adam(self.NN.parameters(), lr=self.lr)`). You might need to re-instantiate it if the learning rate changes via `set_lr`, or use learning rate schedulers.
        *   Define loss functions: `value_loss_fn = torch.nn.MSELoss()`, `policy_loss_fn = torch.nn.CrossEntropyLoss()` (or `KLDivLoss` depending on how probabilities were handled).
        *   Start the training loop (likely just one epoch per call to `update_model` as before):
            *   Convert the NumPy batch (`states`, `targets`, `probs`) to `torch.Tensor`. Ensure `states` has the shape `(batch, channels, height, width)`. `targets` should be `(batch, 1)` or `(batch,)` for MSELoss. `probs` should be `(batch, action_size)` for CrossEntropyLoss.
            *   Move tensors to the appropriate device (e.g., `states = states.to(device)`). Define `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")` earlier.
            *   Set the model to training mode: `self.NN.train()`.
            *   Zero gradients: `optimizer.zero_grad()`.
            *   Perform the forward pass: `pred_value, pred_probs = self.NN(states)` (or just `pred_value` if `q_learning_only`).
            *   Calculate loss: `loss = value_loss_fn(pred_value, targets)` (add policy loss if applicable: `+ policy_loss_fn(pred_probs, target_probs)`).
            *   Backpropagate: `loss.backward()`.
            *   Update weights: `optimizer.step()`.

6.  **Adapt Inference Calls (MCTS, Evaluation):**
    *   Modify `TDAgent.evaluate`, `TDAgent.get_move`, and `mcts.MCTS.search` (specifically the NN prediction part).
    *   Where `self.NN.predict(features)` or `self.nnet_model.predict(features)` was called:
        *   Ensure the input `features` (NumPy array) has the `channels_first` shape `(channels, height, width)`.
        *   Convert features to a PyTorch tensor: `features_tensor = torch.from_numpy(features).float()`.
        *   Add a batch dimension: `features_tensor = features_tensor.unsqueeze(0)`.
        *   Move to the correct device: `features_tensor = features_tensor.to(device)`.
        *   Set the model to evaluation mode: `self.NN.eval()` or `self.nnet_model.eval()`.
        *   Perform inference without calculating gradients:
            ```python
            with torch.no_grad():
                value_pred, policy_pred = self.NN(features_tensor) # Or just value_pred
            ```
        *   Convert results back to NumPy arrays for use in the existing logic:
            *   `value = value_pred.squeeze(0).cpu().numpy()`
            *   `policy = policy_pred.squeeze(0).cpu().numpy()`
        *   Handle the case where only value is returned if `q_learning_only` is True.

7.  **Update Model Loading/Saving:**
    *   In `TDAgent.__init__` and potentially `self_play` (after training):
        *   Replace `keras.models.load_model(filepath)` with:
            ```python
            self.NN = ResNN(...) # Instantiate your PyTorch model class
            self.NN.load_state_dict(torch.load(filepath))
            self.NN.to(device) # Move model to appropriate device
            self.NN.eval() # Set to evaluation mode initially
            ```
    *   In `TDAgent.save_model`:
        *   Replace `self.NN.save(filepath)` with `torch.save(self.NN.state_dict(), filepath)`.
    *   Remove the `load_weights`/`save_weights` methods if they only called Keras equivalents, or adapt them for `state_dict`.

**Phase 3: Multiprocessing and Cleanup**

8.  **Adapt Multiprocessing (`TDAgent.self_play_game_player`):**
    *   Remove any TensorFlow/Keras session setup within `self_play_game_player`.
    *   Ensure the `model_filename` passed correctly refers to the PyTorch `.pth` file.
    *   Inside the function (run by each worker process):
        *   Instantiate the PyTorch `ResNN` model: `model = ResNN(...)`.
        *   Load the state dict: `model.load_state_dict(torch.load(model_filename))`.
        *   Set the device (likely CPU for worker processes unless dedicated GPUs are available): `device = torch.device("cpu")`.
        *   Move the model to the device: `model.to(device)`.
        *   Set to evaluation mode: `model.eval()`.
        *   Pass this loaded `model` instance to the `MCTS` constructor.

9.  **Cleanup and Testing:**
    *   Search the entire project for any remaining `import tensorflow` or `import keras` statements and remove them.
    *   Remove any code related to TensorFlow session management (`tf.ConfigProto`, `keras.backend.set_session`, `keras.backend.clear_session`).
    *   Run a linter (like `flake8`) and formatter (like `black`) over the code.
    *   **Thoroughly Test:**
        *   Test the `ResNN` forward pass with dummy data of the correct shape `(batch, channels, height, width)`.
        *   Test the `extract_features` function output shape.
        *   Test the training loop (`update_model`) with a small batch.
        *   Test inference calls within MCTS.
        *   Run the full `checkersTrainer.py` script for a few iterations to check for runtime errors in self-play, training, and evaluation against the minimax agent. Debug any shape mismatches, device errors, or other issues. 