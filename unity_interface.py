import torch

class LatentSpace:
    def __init__(self, fname):
        self.model = torch.load(fname)

    def forward(self, state):
        state = torch.tensor(state, dtype=torch.float32)
        state = torch.Tensor(state.flatten()).unsqueeze(0)
        latent = self.model.encoder(state)
        return latent.squeeze().detach().numpy()
    
# Usage example:
# test = LatentSpace('levels/level1.pkl')
# print(test.forward([0, 0, 0]))