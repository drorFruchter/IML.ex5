import torch
import torch.nn as nn
from dataset import EuropeDataset

def normalize_tensor(tensor, d):
    mean = torch.mean(tensor, dim=d, keepdim=True)
    std = torch.std(tensor, dim=d, keepdim=True)
    normalized = (tensor - mean) / std
    return normalized

class GMM(nn.Module):
    def __init__(self, n_components):
        super().__init__()        
        self.n_components = n_components
        self.weights = nn.Parameter(torch.randn(n_components))
        self.means = nn.Parameter(torch.randn(n_components, 2))
        self.log_variances = nn.Parameter(torch.zeros(n_components, 2))

    def forward(self, X):
        log_weights = torch.log_softmax(self.weights, dim=0)
        log_likelihoods = []
        for k in range(self.n_components):
            mean = self.means[k]
            log_var = self.log_variances[k]
            var = torch.exp(log_var)
            diff = X - mean
            exponent = -0.5 * torch.sum(diff * diff / var, dim=1)
            norm = -0.5 * torch.sum(log_var) - torch.log(torch.tensor(2 * torch.pi))
            log_likelihood = norm + exponent
            log_likelihoods.append(log_likelihood)
        log_likelihoods = torch.stack(log_likelihoods, dim=1)
        log_likelihood = torch.logsumexp(log_weights + log_likelihoods, dim=1)
        return log_likelihood

    def loss_function(self, log_likelihood):
        return -torch.mean(log_likelihood)

    def sample(self, n_samples):
        weights = torch.softmax(self.weights, dim=0)
        indices = torch.multinomial(weights, n_samples, replacement=True)
        samples = []
        for idx in indices:
            mean = self.means[idx]
            log_var = self.log_variances[idx]
            var = torch.exp(log_var)
            sample = torch.normal(mean, torch.sqrt(var))
            samples.append(sample)
        return torch.stack(samples)

    def conditional_sample(self, n_samples, label):
        mean = self.means[label]
        log_var = self.log_variances[label]
        var = torch.exp(log_var)
        samples = torch.normal(mean, torch.sqrt(var), size=(n_samples, 2))
        return samples

class UMM(nn.Module):
    def __init__(self, n_components):
        super().__init__()        
        self.n_components = n_components
        self.weights = nn.Parameter(torch.randn(n_components))
        self.centers = nn.Parameter(torch.randn(n_components, 2))
        self.log_sizes = nn.Parameter(torch.log(torch.ones(n_components, 2) + torch.rand(n_components, 2)*0.2))

    def forward(self, X):
        log_weights = torch.log_softmax(self.weights, dim=0)
        log_likelihoods = []
        for k in range(self.n_components):
            center = self.centers[k]
            size = torch.exp(self.log_sizes[k])
            lower_bound = center - size / 2
            upper_bound = center + size / 2
            in_bounds = torch.all((X >= lower_bound) & (X <= upper_bound), dim=1)
            log_likelihood = torch.where(in_bounds, -torch.sum(self.log_sizes[k]), torch.tensor(-1e6))
            log_likelihoods.append(log_likelihood)
        log_likelihoods = torch.stack(log_likelihoods, dim=1)
        log_likelihood = torch.logsumexp(log_weights + log_likelihoods, dim=1)
        return log_likelihood

    def loss_function(self, log_likelihood):
        return -torch.mean(log_likelihood)

    def sample(self, n_samples):
        weights = torch.softmax(self.weights, dim=0)
        indices = torch.multinomial(weights, n_samples, replacement=True)
        samples = []
        for idx in indices:
            center = self.centers[idx]
            size = torch.exp(self.log_sizes[idx])
            sample = torch.rand(2) * size + (center - size / 2)
            samples.append(sample)
        return torch.stack(samples)

    def conditional_sample(self, n_samples, label):
        center = self.centers[label]
        size = torch.exp(self.log_sizes[label])
        samples = torch.rand(n_samples, 2) * size + (center - size / 2)
        return samples

if __name__ == "__main__":
    torch.manual_seed(42)
    train_dataset = EuropeDataset('train.csv')
    test_dataset = EuropeDataset('test.csv')

    batch_size = 4096
    num_epochs = 50
    train_dataset.features = normalize_tensor(train_dataset.features, d=0)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_dataset.features = normalize_tensor(test_dataset.features, d=0)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    gmm = GMM(n_components=5)
    optimizer = torch.optim.Adam(gmm.parameters(), lr=0.01)

    for epoch in range(num_epochs):
        for batch in train_loader:
            features, _ = batch
            log_likelihood = gmm(features)
            loss = gmm.loss_function(log_likelihood)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")


        import math
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from dataset import DataHandler

class NewGELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

class CausalSelfAttention(nn.Module):
    def __init__(self, n_head, n_embd, block_size):
        super().__init__()
        self.n_head = n_head
        self.n_embd = n_embd
        self.block_size = block_size
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size))

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        att = torch.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y

class Block(nn.Module):
    def __init__(self, n_head, n_embd, block_size):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_head, n_embd, block_size)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = nn.ModuleDict(dict(
            c_fc    = nn.Linear(n_embd, 4 * n_embd),
            c_proj  = nn.Linear(4 * n_embd, n_embd),
            act     = NewGELU(),
        ))
        m = self.mlp
        self.mlpf = lambda x: m.c_proj(m.act(m.c_fc(x)))

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlpf(self.ln_2(x))
        return x

class GPT(nn.Module):
    def __init__(self, n_layer, n_head, n_embd, vocab_size, block_size):
        super().__init__()
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(vocab_size, self.n_embd),
            wpe = nn.Embedding(block_size, self.n_embd),            
            h = nn.ModuleList([Block(n_head, n_embd, block_size) for _ in range(self.n_layer)]),
            ln_f = nn.LayerNorm(self.n_embd),
        ))
        self.lm_head = nn.Linear(self.n_embd, self.vocab_size, bias=False)

    def forward(self, idx):
        device = idx.device
        b, t = idx.size()
        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        return logits

def train_model(
        train_path,
        test_path=None,
        model=None,                        
        block_size=10,
        n_layer=3,
        n_head=3,
        n_embd=48,
        learning_rate=3e-4,
        batch_size=64,
        epochs=10
):            
    data_handler = DataHandler(train_path, test_path, block_size)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vocab_size = data_handler.get_vocab_size()
    if model is None:
        model = GPT(n_layer, n_head, n_embd, vocab_size, block_size)

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)
    criterion = nn.CrossEntropyLoss()
    print('Using device:', device)

    trainset = data_handler.get_dataset('train')
    testset = data_handler.get_dataset('test')
    
    train_loader = DataLoader(
        trainset,
        sampler=torch.utils.data.RandomSampler(trainset, replacement=True, num_samples=int(1e5)),
        shuffle=False,
        pin_memory=True,
        batch_size=batch_size,        
    )     
    if testset:       
        test_loader = DataLoader(
            testset,
            sampler=torch.utils.data.RandomSampler(testset, replacement=False, num_samples=int(1e4)),
            shuffle=False,
            pin_memory=True,
            batch_size=batch_size,            
        )

    for ep in range(epochs):
        model.train()
        for i, batch in enumerate(tqdm(train_loader)):            
            x, y = batch
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits.view(-1, vocab_size), y.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        with torch.no_grad():
            model.eval()
            total_loss = 0
            correct = 0
            total = 0
            for i, batch in enumerate(tqdm(test_loader)):
                x, y = batch
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits.view(-1, vocab_size), y.view(-1))
                total_loss += loss.item()
                preds = torch.argmax(logits, dim=-1)
                correct += (preds == y).sum().item()
                total += y.numel()
            print(f"Epoch {ep+1}, Test Loss: {total_loss/len(test_loader)}, Accuracy: {correct/total}")

            # Generate sentences
            sentence = "the "
            for i in range(3):
                new_sentence = sentence
                for _ in range(20):
                    tokens = torch.tensor(data_handler.encoder(new_sentence[-block_size:]))[None].to(device)
                    logits = model(tokens)
                    probs = torch.softmax(logits[:, -1, :], dim=-1)
                    next_char = torch.multinomial(probs, num_samples=1).item()
                    new_sentence += data_handler.decoder([next_char])
                print(f"Generated Sentence {i+1}: {new_sentence}")

            # Top-k sampling
            for i in range(3):
                new_sentence = sentence
                for _ in range(20):
                    tokens = torch.tensor(data_handler.encoder(new_sentence[-block_size:]))[None].to(device)
                    logits = model(tokens)
                    probs = torch.softmax(logits[:, -1, :], dim=-1)
                    top_k = 5
                    top_k_probs, top_k_indices = torch.topk(probs, top_k)
                    top_k_probs = top_k_probs / top_k_probs.sum()
                    next_char = torch.multinomial(top_k_probs, num_samples=1).item()
                    new_sentence += data_handler.decoder([top_k_indices[next_char]])
                print(f"Top-k Generated Sentence {i+1}: {new_sentence}")

if __name__=="__main__":
    torch.manual_seed(42)
    train_model('train.txt', 'test.txt')