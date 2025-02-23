import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn.preprocessing import MinMaxScaler
import random
import math

# ------------------- 1. Load and preprocess data -------------------
df = pd.read_csv("/Users/chenxuancao/Library/Mobile Documents/com~apple~CloudDocs/code Source/Python3/lstm_test_emotioncircle/test_date1.csv")

# To prevent CSV column names or order from being inconsistent, unify naming
df.columns = ['delay', 'arousal', 'valence', 'stimulation']

# Sort by delay in ascending order
df.sort_values('delay', inplace=True)
df.reset_index(drop=True, inplace=True)

# Features and targets
features = df[['delay', 'arousal', 'valence', 'stimulation']].values
targets  = df[['arousal', 'valence', 'stimulation']].values

# Normalization
scaler_X = MinMaxScaler()
scaler_Y = MinMaxScaler()

features_scaled = scaler_X.fit_transform(features)
targets_scaled = scaler_Y.fit_transform(targets)

# Rounding is only for demonstration, you can omit it in practice
features_scaled = np.round(features_scaled, decimals=3)
targets_scaled  = np.round(targets_scaled, decimals=3)

# Construct sequential data using a sliding window
seq_length = 4
X_seq = []
Y_seq = []
for i in range(len(features_scaled) - seq_length):
    X_seq.append(features_scaled[i : i+seq_length])
    Y_seq.append(targets_scaled[i+seq_length])

X_seq = torch.tensor(X_seq, dtype=torch.float32)  # (num_samples, seq_len, input_size=4)
Y_seq = torch.tensor(Y_seq, dtype=torch.float32)  # (num_samples, output_size=3)

# ------------------- 2. Define the bidirectional LSTM model -------------------
class BiLSTMPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, noise_std=0.01):
        super(BiLSTMPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.noise_std = noise_std
        
        # bidirectional=True indicates a bidirectional LSTM
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                            batch_first=True, bidirectional=True)
        # For bidirectional, the final output dimension is 2 * hidden_size
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        """
        x shape: (batch_size, seq_length, input_size)
        """
        # ------ Add a bit of noise in the forward pass ------
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(x) * self.noise_std
            x = x + noise
        
        # lstm_out: (batch_size, seq_length, 2*hidden_size)
        lstm_out, _ = self.lstm(x)
        
        # Take the output of the last time step and pass it through the fully connected layer
        out = self.fc(lstm_out[:, -1, :])  # (batch_size, output_size)
        return out

# ------------------- 3. Build the model, define loss and optimizer -------------------
input_size = 4
hidden_size = 16
output_size = 3
num_layers = 1
noise_std = 0.02  # Noise level during training, adjust as needed

model = BiLSTMPredictor(input_size, hidden_size, output_size,
                        num_layers=num_layers, noise_std=noise_std)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ------------------- 4. Train the model (example) -------------------
num_epochs = 50
batch_size = 16

# Simple batch function
def get_batches(X, Y, batch_size):
    # Here we simply split in order, in practice you can shuffle
    for i in range(0, len(X), batch_size):
        x_batch = X[i:i+batch_size]
        y_batch = Y[i:i+batch_size]
        yield x_batch, y_batch

model.train()
for epoch in range(num_epochs):
    epoch_loss = 0.0
    for x_batch, y_batch in get_batches(X_seq, Y_seq, batch_size):
        optimizer.zero_grad()
        output = model(x_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * len(x_batch)
    
    epoch_loss /= len(X_seq)
    if (epoch+1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.6f}")

# ------------------- 5. Inference -------------------
model.eval()
with torch.no_grad():
    # Here we simply perform inference on the entire X_seq
    predictions = model(X_seq).cpu().numpy()  # (num_samples, 3)
    
# Inverse transform the predictions and ground truth
predictions_inv = scaler_Y.inverse_transform(predictions)  # (arousal, valence, stimulation)
targets_inv = scaler_Y.inverse_transform(Y_seq.numpy())

# Because training uses the "next time step" as the target, if you want to align with the original timeline,
# you can offset them as needed. For simplicity, we directly consider
# all points after the first seq_length as the prediction targets.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# ---------------------------------------------------------------------
# Suppose the inverse-transformed prediction results are predictions_inv (N×3): [arousal, valence, stimulation]
# Here we use random values to simulate
N = 1000
arousal_raw  = np.random.uniform(-1.5, 1.5, N)  # random range that can exceed [-1,1]
valence_raw  = np.random.uniform(-1.5, 1.5, N)
stimulation  = np.random.uniform(0, 1,    N)

# ------------------ 1. Clamp (valence, arousal) to the circle of radius 1 within [-1,1] ------------------
# If your data is already within [-1,1], no further scaling is required.
# If radius > 1, scale it down to the unit circle.
def clamp_to_unit_circle(v, a):
    r = np.sqrt(v*v + a*a)
    if r > 1.0:
        v /= r  # scale to radius=1
        a /= r
    return v, a

valence_pred = []
arousal_pred = []
for v0, a0 in zip(valence_raw, arousal_raw):
    v_clamped, a_clamped = clamp_to_unit_circle(v0, a0)
    valence_pred.append(v_clamped)
    arousal_pred.append(a_clamped)

valence_pred = np.array(valence_pred)
arousal_pred = np.array(arousal_pred)

# ------------------ 2. Prepare emoticon images ------------------
emoji_happy   = mpimg.imread("/Users/chenxuancao/Library/Mobile Documents/com~apple~CloudDocs/code Source/Python3/lstm_test_emotioncircle/image/emoji_q1.png")  # Change to your own path
emoji_angry   = mpimg.imread("/Users/chenxuancao/Library/Mobile Documents/com~apple~CloudDocs/code Source/Python3/lstm_test_emotioncircle/image/emoji_q2.png")
emoji_sad     = mpimg.imread("/Users/chenxuancao/Library/Mobile Documents/com~apple~CloudDocs/code Source/Python3/lstm_test_emotioncircle/image/emoji_q3.png")
emoji_excited = mpimg.imread("//Users/chenxuancao/Library/Mobile Documents/com~apple~CloudDocs/code Source/Python3/lstm_test_emotioncircle/image/emoji_q4.png")

def get_emoji_image(a, v):
    """Return the corresponding image array based on the quadrant of (arousal=a, valence=v)."""
    if v >= 0 and a >= 0:
        return emoji_excited  # 1st quadrant
    elif v < 0 and a >= 0:
        return emoji_angry    # 2nd quadrant
    elif v < 0 and a < 0:
        return emoji_sad      # 3rd quadrant
    else:
        return emoji_happy    # 4th quadrant

# ------------------ 3. Create the plot and draw the emotion circle structure ------------------
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_aspect('equal', adjustable='box')
ax.set_xlabel("Valence")
ax.set_ylabel("Arousal")
ax.set_title("Emotion Circle (Unit Circle)")

# (a) Draw the unit circle
circle = plt.Circle((0,0), 1.0, fill=False, color='gray', linestyle='--', linewidth=1.5)
ax.add_artist(circle)

# (b) Draw the axes (cross)
ax.axhline(0, color='black', linewidth=1)
ax.axvline(0, color='black', linewidth=1)

# (c) Trajectory line & current point
path_line, = ax.plot([], [], 'b-', linewidth=1)
point, = ax.plot([], [], 'ro', markersize=5)
xdata, ydata = [], []

# (d) Display the stimulation in the bottom right corner
text_stim = ax.text(0.95, 0.05, '',
                    transform=ax.transAxes,
                    ha='right', va='bottom')

# (e) Use AnnotationBbox to display emoticon images
#     First create a default ab
default_img = OffsetImage(emoji_happy, zoom=0.1)  # zoom can be adjusted
ab = AnnotationBbox(default_img, (0,0), frameon=False)
ax.add_artist(ab)

# ------------------ 4. Initialization/update functions ------------------
def init():
    path_line.set_data([], [])
    point.set_data([], [])
    xdata.clear()
    ydata.clear()
    text_stim.set_text('')
    
    # Reset ab to (0,0) and provide a default image
    ab.xybox = (0, 0)
    ab.offsetbox = OffsetImage(emoji_happy, zoom=0.1)
    return path_line, point, ab, text_stim

def update(frame):
    v = valence_pred[frame]
    a = arousal_pred[frame]
    s = stimulation[frame]
    
    # Update trajectory
    xdata.append(v)
    ydata.append(a)
    path_line.set_data(xdata, ydata)

    # Update the current point (small red dot)
    point.set_data([v], [a])

    # Switch emoticon PNG
    current_emoji_arr = get_emoji_image(a, v)
    ab.xybox = (v, a)  # Move the image center to (v, a)
    ab.offsetbox = OffsetImage(current_emoji_arr, zoom=0.1)

    # Display stimulation in the bottom right corner
    text_stim.set_text(f"stimulation: {s:.3f}")

    return path_line, point, ab, text_stim

# ------------------ 5. Create animation and display it ------------------
ani = animation.FuncAnimation(
    fig, update, frames=N,
    init_func=init, blit=False, interval=400,
)

plt.show()

# If you want to save the animation:
# ani.save("emotion_circle.gif", writer='pillow')
