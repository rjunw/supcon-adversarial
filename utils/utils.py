import torch
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def get_logits(model, loader):
    logits = []
    y_test = []
    for x, y, z in tqdm(loader):
        x = [x[0].to(device), x[1].to(device)]
        logits.append(model(x[0]).detach().cpu())
        y_test.append(y.detach().cpu())
    return torch.cat(logits).numpy(), torch.cat(y_test).numpy()