from torch_hyperbolic.models.hgnn_film import HFiLM
from torch_geometric.datasets import Planetoid
import torch
from sklearn.metrics import balanced_accuracy_score

torch.set_default_dtype(torch.float64)

def get_accuracy(out, truth, mask):
    return balanced_accuracy_score(truth[mask], out.argmax(dim=1)[mask])

dataset = Planetoid(root='/tmp/Cora', name='Cora')

input_dim = dataset.num_node_features

output_dim = dataset.num_classes
hidden_dim = 12
loss_function = torch.nn.BCEWithLogitsLoss(reduction="none")
model = HFiLM(in_channels=input_dim, out_channels=output_dim, num_relations=2, hidden_dim=hidden_dim, manifold="PoincareBall", gcn_kwargs={"c_per_relation": True}).double()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

y_onehot = torch.FloatTensor(dataset.y.shape[0], output_dim)
y_onehot.zero_()
y_onehot.scatter_(1, dataset.y.unsqueeze(-1), 1)

edge_index = torch.cat((dataset.edge_index, dataset.edge_index), dim=1)
edge_type = torch.cat((torch.zeros_like(dataset.edge_index)[0, :], torch.ones_like(dataset.edge_index)[0, :]), dim=0)
epochs = 100

print("Model curvatures before training: {}".format([round(x.detach().item(), 3) for x in model.curvatures]))
print("First HFiLM Layer curvatures before training: {}".format([round(x.detach().item(), 3) for x in model.gnn1.curvatures]))
print("Second HFiLM Layer curvatures before training: {}".format([round(x.detach().item(), 3) for x in model.gnn2.curvatures]))

for epoch in range(epochs):
    # train
    model.zero_grad()
    out = model(dataset.x.double(), edge_index, edge_type)
    loss = loss_function(out[dataset.train_mask].squeeze(), y_onehot[dataset.train_mask].squeeze()).squeeze()
    loss = loss.mean()
    loss.backward()
    optimizer.step()

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
    print("Model curvatures after training: {}".format([round(x.detach().item(), 3) for x in model.curvatures]))
    print("First HFiLM Layer curvatures after training: {}".format([round(x.detach().item(), 3) for x in model.gnn1.curvatures]))
    print("Second HFiLM Layer curvatures after training: {}".format([round(x.detach().item(), 3) for x in model.gnn2.curvatures]))