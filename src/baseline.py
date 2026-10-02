import torch, torch.nn as nn, numpy as np

class TwoLayerNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.h, self.out = nn.Linear(500, 128), nn.Linear(128, 4)
    def forward(self, x):
        z = torch.tanh(self.h(x))
        return z, self.out(z)          # (embedding, logits)

def train_classifier(model, data, epochs=300, lr=1e-4, bs=512, patience=30, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    Xtr, ytr = torch.tensor(data["train"][0]), torch.tensor(data["train"][1])
    Xdv, ydv = torch.tensor(data["dev"][0]), torch.tensor(data["dev"][1])
    opt, ce = torch.optim.Adam(model.parameters(), lr=lr), nn.CrossEntropyLoss()
    best, best_state, bad, hist = -1.0, None, 0, []
    for ep in range(epochs):
        model.train(); perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), bs):
            b = perm[i:i + bs]
            opt.zero_grad(); ce(model(Xtr[b])[1], ytr[b]).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            tr_acc = (model(Xtr)[1].argmax(1) == ytr).float().mean().item()
            dv_acc = (model(Xdv)[1].argmax(1) == ydv).float().mean().item()
        hist.append((tr_acc, dv_acc))
        if dv_acc > best:
            best, bad = dv_acc, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience: break
    model.load_state_dict(best_state)
    return model, hist
