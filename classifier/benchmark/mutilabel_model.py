import torch

batch_size = 2
num_classes = 11

loss_fn = torch.nn.BCELoss()

outputs_before_sigmoid = torch.randn(batch_size, num_classes)
sigmoid_outputs = torch.sigmoid(outputs_before_sigmoid)
target_classes = torch.randint(0, 2, (batch_size, num_classes))  # randints in [0, 2).

loss = loss_fn(sigmoid_outputs, target_classes)

# alternatively, use BCE with logits, on outputs before sigmoid.
loss_fn_2 = torch.nn.BCEWithLogitsLoss()
loss2 = loss_fn_2(outputs_before_sigmoid, target_classes)
assert loss == loss2

# def multilabel():
    # labels = torch.tensor([1, 4, 1, 0, 5, 2])
    # labels = labels.unsqueeze(0)
    # target = torch.zeros(labels.size(0), 15).scatter_(1, labels, 1.)
    # print(target)
    # tensor([[1., 1., 1., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])
    # nn.BCEWithLogitsLoss takes the raw logits of your model (without any non-linearity) and applies the sigmoid internally.
    # If you would like to add the sigmoid activation to your model, you should use nn.BCELoss instead.
    # https://discuss.pytorch.org/t/is-there-an-example-for-multi-class-multilabel-classification-in-pytorch/53579/11
#     model = nn.Linear(20, 5) # predict logits for 5 classes
#     x = torch.randn(1, 20)
#     y = torch.tensor([[1., 0., 1., 0., 0.]]) # get classA and classC as active

#     criterion = nn.BCEWithLogitsLoss()
#     optimizer = optim.SGD(model.parameters(), lr=1e-1)

#     for epoch in range(20):
#         optimizer.zero_grad()
#         output = model(x)
#         loss = criterion(output, y)
#         loss.backward()
#         optimizer.step()
#         print('Loss: {:.3f}'.format(loss.item()))