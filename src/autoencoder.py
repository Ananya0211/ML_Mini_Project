import torch
import torch.nn as nn
import numpy as np

class VanillaAutoencoder(nn.Module):
    """
    Vanilla Autoencoder:
    Encoder: 500 -> 256 -> 192 -> 128 -> 64 (tanh activations)
    Decoder: 64 -> 128 -> 192 -> 256 -> 500 (tanh activations)
    """
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(500, 256), nn.Tanh(),
            nn.Linear(256, 192), nn.Tanh(),
            nn.Linear(192, 128), nn.Tanh(),
            nn.Linear(128, 64),  nn.Tanh()
        )
        self.decoder = nn.Sequential(
            nn.Linear(64, 128),  nn.Tanh(),
            nn.Linear(128, 192), nn.Tanh(),
            nn.Linear(192, 256), nn.Tanh(),
            nn.Linear(256, 500)
        )

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        # Returns (z, logits) interface compatibility: logits is None for pure AE
        return z, recon


class DeepSoftmaxAutoencoder(nn.Module):
    """
    Deep Softmax Autoencoder (Sawhney et al., 2018):
    Autoencoder Backbone: 500 -> 256 -> 192 -> 128 -> 64 -> 128 -> 192 -> 256 -> 500
    Classifier Head: 64 -> 32 -> 16 -> 4 attached to 64-dim bottleneck
    """
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(500, 256), nn.Tanh(),
            nn.Linear(256, 192), nn.Tanh(),
            nn.Linear(192, 128), nn.Tanh(),
            nn.Linear(128, 64),  nn.Tanh()
        )
        self.decoder = nn.Sequential(
            nn.Linear(64, 128),  nn.Tanh(),
            nn.Linear(128, 192), nn.Tanh(),
            nn.Linear(192, 256), nn.Tanh(),
            nn.Linear(256, 500)
        )
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),   nn.Tanh(),
            nn.Linear(32, 16),   nn.Tanh(),
            nn.Linear(16, 4)
        )

    def forward(self, x):
        z = self.encoder(x)
        logits = self.classifier(z)
        recon = self.decoder(z)
        return z, logits, recon


def train_deep_softmax_ae(model, data, gamma=0.9, epochs=300, lr=1e-4, bs=512, patience=30, seed=0):
    """
    Trains Deep Softmax Autoencoder with joint loss:
    Loss = gamma * MSE(recon, x) + (1 - gamma) * CrossEntropy(logits, y)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    Xtr, ytr = torch.tensor(data["train"][0]), torch.tensor(data["train"][1])
    Xdv, ydv = torch.tensor(data["dev"][0]), torch.tensor(data["dev"][1])

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()
    ce = nn.CrossEntropyLoss()

    best_acc, best_state, bad, hist = -1.0, None, 0, []

    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), bs):
            b = perm[i:i + bs]
            opt.zero_grad()
            z, logits, recon = model(Xtr[b])
            loss = gamma * mse(recon, Xtr[b]) + (1.0 - gamma) * ce(logits, ytr[b])
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            tr_z, tr_logits, tr_recon = model(Xtr)
            dv_z, dv_logits, dv_recon = model(Xdv)

            tr_acc = (tr_logits.argmax(1) == ytr).float().mean().item()
            dv_acc = (dv_logits.argmax(1) == ydv).float().mean().item()
            dv_loss = (gamma * mse(dv_recon, Xdv) + (1.0 - gamma) * ce(dv_logits, ydv)).item()

        hist.append((tr_acc, dv_acc, dv_loss))

        if dv_acc > best_acc:
            best_acc = dv_acc
            bad = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, hist


class EvaluateAdapter(nn.Module):
    """Wrapper around DeepSoftmaxAutoencoder returning (embedding, logits) for evaluate.py interface."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        z, logits, _ = self.model(x)
        return z, logits
