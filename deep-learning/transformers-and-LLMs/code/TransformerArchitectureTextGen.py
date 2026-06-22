import math
import random
import torch
import torch.nn as nn
import torch.optim as optim

# ==========================================
# 1. THE CORPUS GENERATOR (THE MINI-BOOK)
# ==========================================
characters = ["The knight", "The old wizard", "The angry king", "The fierce dragon", "A clever thief", "The brave queen"]
actions = ["walked to", "ran towards", "flew over", "silently attacked", "bravely defended", "secretly explored"]
locations = ["the dark castle", "the tall snowy mountain", "the deep green forest", "the quiet village", "the deep cave"]
endings = ["with great courage.", "in the middle of the night.", "and found a chest of gold.", "and cast a powerful spell.", "but was chased away.", "and rested by the fire."]

print("Drafting the Mini-Book corpus...")
sentences = []
for _ in range(5000): 
    sentences.append(f"{random.choice(characters)} {random.choice(actions)} {random.choice(locations)} {random.choice(endings)}")

corpus = " ".join(sentences)
print(f"Corpus generated! Total characters: {len(corpus):,}\n")


# ==========================================
# 2. TOKENIZATION & BATCHING
# ==========================================
vocab = sorted(list(set(corpus)))
vocab_size = len(vocab)
stoi = {ch: i for i, ch in enumerate(vocab)}
itos = {i: ch for i, ch in enumerate(vocab)}

encoded_book = torch.tensor([stoi[c] for c in corpus], dtype=torch.long)

max_len = 64 # Shortened slightly for faster Seq2Seq dual-processing
batch_size = 32
device = torch.device('xpu' if hasattr(torch, 'xpu') and torch.xpu.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')

def get_gpt_batch():
    ix = torch.randint(len(encoded_book) - max_len - 1, (batch_size,))
    x = torch.stack([encoded_book[i : i+max_len] for i in ix])
    y = torch.stack([encoded_book[i+1 : i+max_len+1] for i in ix])
    return x.to(device), y.to(device)

def get_seq2seq_batch():
    # Source is the first half, Target is the second half of the chunk
    half_len = max_len // 2
    ix = torch.randint(len(encoded_book) - max_len - 1, (batch_size,))
    
    src = torch.stack([encoded_book[i : i+half_len] for i in ix])
    tgt_input = torch.stack([encoded_book[i+half_len : i+max_len] for i in ix])
    tgt_label = torch.stack([encoded_book[i+half_len+1 : i+max_len+1] for i in ix])
    
    return src.to(device), tgt_input.to(device), tgt_label.to(device)


# ==========================================
# 3. THE ARCHITECTURES
# ==========================================

# --- A. MiniGPT (Decoder-Only) ---
class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        assert d_model % nhead == 0
        self.nhead = nhead
        self.d_k = d_model // nhead
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, T, C = x.size() 
        q = self.q_proj(x).view(B, T, self.nhead, self.d_k).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.nhead, self.d_k).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.nhead, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(mask, float('-inf'))
        
        attn_weights = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn_weights, v).transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, nhead)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Linear(dim_feedforward, d_model)
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

class MiniGPT(nn.Module):
    def __init__(self, v_size, d_model=128, nhead=4, dim_feedforward=256, num_layers=2):
        super().__init__()
        self.emb = nn.Embedding(v_size, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        self.blocks = nn.Sequential(*[TransformerBlock(d_model, nhead, dim_feedforward) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, v_size)

    def forward(self, x):
        seq_len = x.size(1)
        pos = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(x.size(0), -1)
        out = self.emb(x) + self.pos(pos)
        out = self.blocks(out)
        out = self.ln_f(out)
        return self.fc(out)


# --- B. MiniSeq2Seq (Encoder-Decoder) ---
class MiniSeq2Seq(nn.Module):
    def __init__(self, v_size, d_model=128, nhead=4, num_layers=2, dim_feedforward=256):
        super().__init__()
        self.emb = nn.Embedding(v_size, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        # Using PyTorch's built-in full transformer block for simplicity in the Seq2Seq mapping
        self.transformer = nn.Transformer(
            d_model=d_model, 
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        self.fc = nn.Linear(d_model, v_size)

    def forward(self, src, tgt):
        seq_len_src = src.size(1)
        seq_len_tgt = tgt.size(1)
        
        pos_src = torch.arange(seq_len_src, device=src.device).unsqueeze(0).expand(src.size(0), -1)
        pos_tgt = torch.arange(seq_len_tgt, device=tgt.device).unsqueeze(0).expand(tgt.size(0), -1)
        
        src_emb = self.emb(src) + self.pos(pos_src)
        tgt_emb = self.emb(tgt) + self.pos(pos_tgt)
        
        # Prevent the decoder from looking ahead
        tgt_mask = self.transformer.generate_square_subsequent_mask(seq_len_tgt, device=tgt.device)
        
        out = self.transformer(src_emb, tgt_emb, tgt_mask=tgt_mask)
        return self.fc(out)


# ==========================================
# 4. TRAINING LOOPS
# ==========================================
steps = 400
criterion = nn.CrossEntropyLoss()

# Train GPT
print(f"--- Training MiniGPT on {device} ---")
gpt_model = MiniGPT(vocab_size).to(device)
gpt_opt = optim.Adam(gpt_model.parameters(), lr=0.001)

for step in range(steps):
    xb, yb = get_gpt_batch()
    gpt_opt.zero_grad()
    logits = gpt_model(xb)
    loss = criterion(logits.view(-1, vocab_size), yb.view(-1))
    loss.backward()
    gpt_opt.step()
    if (step + 1) % 100 == 0: 
        print(f"GPT Step {step+1:03d}/{steps} | Loss: {loss.item():.4f}")

# Train Seq2Seq
print(f"\n--- Training MiniSeq2Seq on {device} ---")
s2s_model = MiniSeq2Seq(vocab_size).to(device)
s2s_opt = optim.Adam(s2s_model.parameters(), lr=0.001)

for step in range(steps):
    src, tgt_input, tgt_label = get_seq2seq_batch()
    s2s_opt.zero_grad()
    logits = s2s_model(src, tgt_input)
    loss = criterion(logits.view(-1, vocab_size), tgt_label.view(-1))
    loss.backward()
    s2s_opt.step()
    if (step + 1) % 100 == 0: 
        print(f"Seq2Seq Step {step+1:03d}/{steps} | Loss: {loss.item():.4f}")


# ==========================================
# 5. CREATIVE TEXT INFERENCE (Strictly Softmax)
# ==========================================
def generate_gpt(model, seed_text, num_new_chars=100):
    model.eval()
    seq = [stoi[c] for c in seed_text]
    
    with torch.no_grad():
        for _ in range(num_new_chars):
            context = seq[-max_len:]
            input_tensor = torch.tensor([context], device=device)
            logits = model(input_tensor)
            next_token_logits = logits[0, -1, :]
            
            # Softmax only, absolutely no argmax
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            seq.append(next_token)
            
    return "".join(itos[i] for i in seq)

def generate_seq2seq(model, source_text, num_new_chars=100):
    model.eval()
    # Crop source text if it exceeds maximum context allowance
    src_seq = [stoi[c] for c in source_text][-max_len//2:]
    src_tensor = torch.tensor([src_seq], device=device)
    
    # Initialize target with a space to kick off generation
    tgt_seq = [stoi[" "]] 
    
    with torch.no_grad():
        for _ in range(num_new_chars):
            tgt_tensor = torch.tensor([tgt_seq[-max_len//2:]], device=device)
            logits = model(src_tensor, tgt_tensor)
            next_token_logits = logits[0, -1, :]
            
            # Softmax only, absolutely no argmax
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            tgt_seq.append(next_token)
            
    return source_text + " -> " + "".join(itos[i] for i in tgt_seq[1:])

# Let's write some stories
seed = "The fierce dragon"
print(f"\n--- AI WRITING SESSIONS ---")
print(f"Seed Text: '{seed}'\n")

print("GPT Continuation:")
print(generate_gpt(gpt_model, seed))

print("\nSeq2Seq Continuation (Input -> Output Mapping):")
print(generate_seq2seq(s2s_model, seed))