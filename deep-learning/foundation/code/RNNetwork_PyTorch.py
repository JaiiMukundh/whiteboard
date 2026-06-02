import torch
import torch.nn as nn
import torch.optim as optim

# =====================================================
# RNN CELL
# =====================================================

class RNNCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()

        combined_dim = input_dim + hidden_dim

        # Only one weight matrix and one bias for a standard RNN
        self.W_h = nn.Parameter(torch.randn(hidden_dim, combined_dim) * 0.1)
        self.b_h = nn.Parameter(torch.zeros(hidden_dim, 1))

    def forward(self, x_t, h_prev):
        # Stack h_prev and x_t into a single column vector
        combined = torch.cat([h_prev, x_t], dim=0)

        # The core RNN math
        h_t = torch.tanh(self.W_h @ combined + self.b_h)

        return h_t


# =====================================================
# CHARACTER RNN
# =====================================================

class CharRNN(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim):
        super().__init__()

        self.hidden_dim = hidden_dim

        self.E = nn.Parameter(torch.randn(emb_dim, vocab_size) * 0.1)

        self.cell = RNNCell(emb_dim, hidden_dim)

        self.W_y = nn.Parameter(torch.randn(vocab_size, hidden_dim) * 0.1)
        self.b_y = nn.Parameter(torch.zeros(vocab_size, 1))

    def embed(self, idx):
        return self.E[:, idx:idx+1]

    def forward(self, seq):

        h0 = torch.zeros(self.hidden_dim, 1)

        x1 = self.embed(seq[0])
        h1 = self.cell(x1, h0)

        x2 = self.embed(seq[1])
        h2 = self.cell(x2, h1)

        x3 = self.embed(seq[2])
        h3 = self.cell(x3, h2)

        x4 = self.embed(seq[3])
        h4 = self.cell(x4, h3)

        x5 = self.embed(seq[4])
        h5 = self.cell(x5, h4)

        logits = self.W_y @ h5 + self.b_y

        return logits
    
    @torch.no_grad()
    def blurt(self, seed, n_chars):

        h0 = torch.zeros(self.hidden_dim, 1)

        x1 = self.embed(seed[0])
        h1 = self.cell(x1, h0)

        x2 = self.embed(seed[1])
        h2 = self.cell(x2, h1)

        x3 = self.embed(seed[2])
        h3 = self.cell(x3, h2)

        x4 = self.embed(seed[3])
        h4 = self.cell(x4, h3)

        x5 = self.embed(seed[4])
        h5 = self.cell(x5, h4)

        h = h5

        generated = []

        for _ in range(n_chars):

            logits = self.W_y @ h + self.b_y

            # 1. Set a temperature
            temperature = 0.8 

            # 2. Flatten the (vocab_size, 1) column vector into a 1D array (vocab_size)
            logits_1d = logits.squeeze()

            # 3. Scale the logits
            scaled_logits = logits_1d / temperature

            # 4. Calculate probabilities along the only dimension left (dim=0)
            probs = torch.softmax(scaled_logits, dim=0)

            # 5. Spin the roulette wheel
            next_idx = torch.multinomial(probs, num_samples=1).item()
            generated.append(next_idx)

            x = self.embed(next_idx)

            h = self.cell(x, h)

        return generated

# =====================================================
# DATASET
# =====================================================

text = "to be or not to be is the greatest boon for mankind's brilliance"

chars = sorted(set(text))
vocab_size = len(chars)

c2i = {c: i for i, c in enumerate(chars)}
i2c = {i: c for i, c in enumerate(chars)}

X = []
Y = []

window = 5

for i in range(len(text) - window):
    X.append([c2i[c] for c in text[i:i + window]])
    Y.append(c2i[text[i + window]])

print(X)
print(Y)

# =====================================================
# MODEL
# =====================================================

model = CharRNN(
    vocab_size=vocab_size,
    emb_dim=16,
    hidden_dim=32
)

cost = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)


# =====================================================
# TRAINING
# =====================================================

epochs = 200

for epoch in range(epochs):

    total_loss = 0

    for seq, target in zip(X, Y):

        optimizer.zero_grad()

        logits = model(seq)

        loss = cost(
            logits.T,                 # shape: (1, vocab_size)
            torch.tensor([target])    # shape: (1,)
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    if (epoch + 1) % 50 == 0:
        avg_loss = total_loss / len(X)

        print(
            f"Epoch {epoch+1:3d}/{epochs} "
            f"| Loss = {avg_loss:.4f}"
        )


# =====================================================
# GENERATION
# =====================================================

seed_str = text[:5]
seed = [c2i[c] for c in seed_str]

generated_indices = model.blurt(seed, 5)
generated_text = "".join(i2c[i] for i in generated_indices)

print("\nSeed:", repr(seed_str))
print("Generated:", repr(generated_text))