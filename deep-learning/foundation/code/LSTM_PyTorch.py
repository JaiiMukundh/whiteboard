import torch
import torch.nn as nn
import torch.optim as optim


# =====================================================
# LSTM CELL (Colah-style equations)
# =====================================================

class LSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()

        combined_dim = input_dim + hidden_dim

        self.W_f = nn.Parameter(torch.randn(hidden_dim, combined_dim) * 0.1)
        self.b_f = nn.Parameter(torch.ones(hidden_dim, 1))

        self.W_i = nn.Parameter(torch.randn(hidden_dim, combined_dim) * 0.1)
        self.b_i = nn.Parameter(torch.zeros(hidden_dim, 1))

        self.W_c = nn.Parameter(torch.randn(hidden_dim, combined_dim) * 0.1)
        self.b_c = nn.Parameter(torch.zeros(hidden_dim, 1))

        self.W_o = nn.Parameter(torch.randn(hidden_dim, combined_dim) * 0.1)
        self.b_o = nn.Parameter(torch.zeros(hidden_dim, 1))

    def forward(self, x_t, h_prev, c_prev):
        combined = torch.cat([h_prev, x_t], dim=0)

        f_t = torch.sigmoid(self.W_f @ combined + self.b_f)
        i_t = torch.sigmoid(self.W_i @ combined + self.b_i)
        c_tilde = torch.tanh(self.W_c @ combined + self.b_c)
        o_t = torch.sigmoid(self.W_o @ combined + self.b_o)

        c_t = f_t * c_prev + i_t * c_tilde
        h_t = o_t * torch.tanh(c_t)

        return h_t, c_t


# =====================================================
# CHARACTER LSTM
# =====================================================

class CharLSTM(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim):
        super().__init__()

        self.hidden_dim = hidden_dim

        self.E = nn.Parameter(torch.randn(emb_dim, vocab_size) * 0.1)

        self.cell = LSTMCell(emb_dim, hidden_dim)

        self.W_y = nn.Parameter(torch.randn(vocab_size, hidden_dim) * 0.1)
        self.b_y = nn.Parameter(torch.zeros(vocab_size, 1))

    def embed(self, idx):
        return self.E[:, idx:idx+1]

    def forward(self, seq):

        h0 = torch.zeros(self.hidden_dim, 1)
        c0 = torch.zeros(self.hidden_dim, 1)

        x1 = self.embed(seq[0])
        h1, c1 = self.cell(x1, h0, c0)

        x2 = self.embed(seq[1])
        h2, c2 = self.cell(x2, h1, c1)

        x3 = self.embed(seq[2])
        h3, c3 = self.cell(x3, h2, c2)

        x4 = self.embed(seq[3])
        h4, c4 = self.cell(x4, h3, c3)

        x5 = self.embed(seq[4])
        h5, c5 = self.cell(x5, h4, c4)

        logits = self.W_y @ h5 + self.b_y

        return logits
    
    @torch.no_grad()
    def blurt(self, seed, n_chars):

        h0 = torch.zeros(self.hidden_dim, 1)
        c0 = torch.zeros(self.hidden_dim, 1)

        x1 = self.embed(seed[0])
        h1, c1 = self.cell(x1, h0, c0)

        x2 = self.embed(seed[1])
        h2, c2 = self.cell(x2, h1, c1)

        x3 = self.embed(seed[2])
        h3, c3 = self.cell(x3, h2, c2)

        x4 = self.embed(seed[3])
        h4, c4 = self.cell(x4, h3, c3)

        x5 = self.embed(seed[4])
        h5, c5 = self.cell(x5, h4, c4)

        h, c = h5, c5

        generated = []

        for _ in range(n_chars):

            logits = self.W_y @ h + self.b_y

            next_idx = torch.argmax(logits).item()
            generated.append(next_idx)

            x = self.embed(next_idx)

            h, c = self.cell(x, h, c)

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


# =====================================================
# MODEL
# =====================================================

model = CharLSTM(
    vocab_size=vocab_size,
    emb_dim=16,
    hidden_dim=32
)

criterion = nn.CrossEntropyLoss()
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

        loss = criterion(
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

generated_indices = model.blurt(seed, 20)
generated_text = "".join(i2c[i] for i in generated_indices)

print("\nSeed:", repr(seed_str))
print("Generated:", repr(generated_text))