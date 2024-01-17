# train.py

import torch

def train_model(model, train_loader, val_loader, loss_fn, optimizer, num_epochs, device):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        average_loss = running_loss / len(train_loader)

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {average_loss:.4f}")

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for val_inputs, val_labels in val_loader:
                val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)

                val_outputs = model(val_inputs)
                val_loss += loss_fn(val_outputs, val_labels).item()

                _, predicted = val_outputs.max(1)
                total += val_labels.size(0)
                correct += predicted.eq(val_labels).sum().item()

        val_accuracy = correct / total
        val_loss /= len(val_loader)

        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

    print("Training finished.")

def evaluate_model(model, test_loader, loss_fn, device):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for test_inputs, test_labels in test_loader:
            test_inputs, test_labels = test_inputs.to(device), test_labels.to(device)

            test_outputs = model(test_inputs)
            test_loss += loss_fn(test_outputs, test_labels).item()

            _, predicted = test_outputs.max(1)
            total += test_labels.size(0)
            correct += predicted.eq(test_labels).sum().item()

    test_accuracy = correct / total
    test_loss /= len(test_loader)

    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

    return test_accuracy, test_loss
