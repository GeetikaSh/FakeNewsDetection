# 🧠 What Is LSTM?
LSTM stands for Long Short-Term Memory, a special kind of Recurrent Neural Network (RNN) designed to remember information over long sequences.

## Problem with Regular RNNs:
- They suffer from vanishing gradients, which means they forget long-term dependencies.
- For example, remembering the context from the beginning of a long sentence is hard for a regular RNN.

## LSTM to the Rescue!
LSTMs fix this with a cell state (like a conveyor belt of memory) and gates that control what to keep, forget, or update.

---

## 🔍 LSTM Cell Mechanics

### 1. 🧹 Forget Gate (f<sub>t</sub>)
Decides what information to throw away from the cell state.

$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$

- σ is the sigmoid function.
- Outputs values between 0 and 1:
  - `0` = completely forget
  - `1` = completely keep

---

### 2. ➕ Input Gate (i<sub>t</sub>) + Candidate Layer (~C<sub>t</sub>)
Decides what new information to add to the cell state.

$$ i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$

$$\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)$$

- `i_t`: decides which parts of candidate values to add.
- `~C_t`: candidate values for updating the cell state.

---

### 3. 📦 Update Cell State

$$C_t = f_t * C_{t-1} + i_t * \tilde{C}_t$$

- Combines what to forget and what to add.
- Old memory is partially forgotten, new memory is added.

---

### 4. 🚪 Output Gate (o<sub>t</sub>)

Decides what to output from the current state.

$$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$

$$h_t = o_t * \tanh(C_t)$$

- `h_t` is the final output passed to the next LSTM cell and used for predictions.

---

## 🛠️ LSTM Summary (Visual Flow)
```
                x_t
                 ↓
        +-----------------+
        |     Forget      |
        |     Gate (f_t)  |
        +-----------------+
                 ↓
        +-----------------+       +------------------+
        |   Input Gate    | + --->|  Candidate (C~)  |
        +-----------------+       +------------------+
                 ↓                      ↓
             Combine: i_t * C~_t       ↓
                 ↓                     ↓
     Update: C_t = f_t * C_{t-1} + i_t * C~_t
                 ↓
        +-----------------+
        |   Output Gate   |
        +-----------------+
                 ↓
             h_t = o_t * tanh(C_t)
```

## 🧪 Why It Works Well for Fake News Detection
Text often has long-term dependencies — words early in a sentence can completely change the meaning of later words.

Example:

"It was reported that the politician did not accept the bribe."

A model must remember "did not" even several words apart to understand the sentence correctly. LSTM does this better than traditional models.
