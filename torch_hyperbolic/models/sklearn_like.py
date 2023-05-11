import torch
import time
from sklearn.metrics import balanced_accuracy_score


class GNNClassifier:
    def __init__(self, model, max_iter=100, *args, **kwargs):
        self.model = model
        self.max_iter = max_iter
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        self.loss_function = torch.nn.BCEWithLogitsLoss(reduction="none")

        self.best_accuracy = 0
        self.model_name = str(time.time()).split(".")[-1]
        
    def fit(self, x, edge_index, y, train_mask, val_mask):
        for epoch in range(self.max_iter):
            # train
            self.model.zero_grad()
            out = self.model(x, edge_index)
            
            loss = self.loss_function(out[train_mask].squeeze(), y[train_mask].squeeze()).squeeze()
            loss = loss.mean()
            loss.backward()
            self.optimizer.step()

            # val
            if (epoch + 1) % 5 == 0:        
                with torch.no_grad():
                    accuracy_train = self.get_accuracy(out, y, train_mask)
                    accuracy_val = self.get_accuracy(out, y, val_mask)
                    val_loss = self.loss_function(out[val_mask].squeeze(), y[val_mask].squeeze()).squeeze().mean()
                print("Epoch: {}, Train Loss: {:.4f}, Val Loss: {:.4f}, Train Acc: {:.4f}, Val Acc: {:.4f}".format(epoch + 1, loss, val_loss, accuracy_train, accuracy_val))
                if accuracy_val > self.best_accuracy:
                    self.best_accuracy = accuracy_val
                    torch.save(self.model, "./models/" + self.model_name + ".pt")

        self.model = torch.load("./models/" + self.model_name + ".pt")

    def transform(self, x, edge_index, mask, y=None, return_accuracy=False):
        with torch.no_grad():
            out = self.model(x, edge_index)
            if return_accuracy:
                return out[mask], self.get_accuracy(out, y, mask)
            else:
                return out[mask]
        
    def fit_transform(self, x, edge_index, y, train_mask, val_mask, test_mask, return_accuracy=False):
        self.fit(x, edge_index, y, train_mask, val_mask)
        return self.transform(x, edge_index, test_mask, y, return_accuracy)

    def get_accuracy(self, out, truth, mask):
        return balanced_accuracy_score(truth.argmax(dim=1)[mask], out.argmax(dim=1)[mask])
