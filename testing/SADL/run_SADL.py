import DSA
import LSA
import helpers
import torch
import numpy as np
import pickle

if __name__ == "__main__":
    model_path = "./models/test/best.pt"
    train_path = "./data/test/train"
    val_path = "./data/test/val"
    
    #TODO fix this up
    model = torch.load(model_path)
    train_loader = torch.utils.data.DataLoader()  # Define your data loader here
    val_loader = torch.utils.data.DataLoader()  # Define your data loader here
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ats = helpers.get_ats(model, train_loader, device)
    lsa_results = LSA.fit_lsa(ats)
    dsa_results = DSA.fetch_dsa(model, train_loader, device)
    
    with open("lsa_results.pkl", "wb") as f:
        pickle.dump(lsa_results, f)
    with open("dsa_results.pkl", "wb") as f:
        pickle.dump(dsa_results, f)
    