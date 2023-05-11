from torch_hyperbolic.models import GNN
import torch_hyperbolic.datasets as th_datasets
import torch_geometric.datasets as datasets
import torch
from sklearn.metrics import balanced_accuracy_score

torch.set_default_dtype(torch.float64)

def get_accuracy(out, truth, mask):
    return balanced_accuracy_score(truth[mask], out.argmax(dim=1)[mask])

#dataset = datasets.Planetoid(root='/tmp/Cora', name='Cora')

dataset = th_datasets.DiseaseDataset()

input_dim = dataset.num_node_features

output_dim = dataset.num_classes
hidden_dim = 3
loss_function = torch.nn.BCEWithLogitsLoss(reduction="none")

model = GNN(in_channels=input_dim, out_channels=output_dim, hidden_dim=hidden_dim,  gcn_kwargs={})
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

y_onehot = torch.FloatTensor(dataset.y.shape[0], output_dim)
y_onehot.zero_()
y_onehot.scatter_(1, dataset.y.unsqueeze(-1), 1)

epochs = 100

for epoch in range(epochs):
    # train
    model.zero_grad()
    out = model(dataset.x, dataset.edge_index)
    
    loss = loss_function(out[dataset.train_mask].squeeze(), y_onehot[dataset.train_mask].squeeze()).squeeze()
    loss = loss.mean()
    loss.backward()
    optimizer.step()

    # val
    if (epoch + 1) % 5 == 0:        
        with torch.no_grad():
            accuracy_train = get_accuracy(out, dataset.y, dataset.train_mask)
            accuracy_val = get_accuracy(out, dataset.y, dataset.val_mask)
            val_loss = loss_function(out[dataset.val_mask].squeeze(), y_onehot[dataset.val_mask].squeeze()).squeeze().mean()
        print("Epoch: {}, Train Loss: {:.4f}, Val Loss: {:.4f}, Train Acc: {:.4f}, Val Acc: {:.4f}".format(epoch + 1, loss, val_loss, accuracy_train, accuracy_val))

    

# test
with torch.no_grad():
    test_loss = loss_function(out[dataset.test_mask].squeeze(), y_onehot[dataset.test_mask].squeeze()).squeeze().mean()
    accuracy_test = get_accuracy(out, dataset.y, dataset.test_mask)
    print("Test Loss: {:.4f}, Test Acc: {:.4f}".format(test_loss, accuracy_test))