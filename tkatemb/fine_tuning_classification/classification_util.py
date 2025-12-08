import torch

def calc_accuracy_loader(data_loader, model, device, num_batches=None):
    model.eval()
    correct_predictions, num_examples = 0, 0
    
    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)
            
            with torch.no_grad():
                logits = model(input_batch)[:, -1, :]  # 1
            
            predicted_labels = torch.argmax(logits, dim=-1)
            num_examples += predicted_labels.shape[0]
            correct_predictions += (predicted_labels == target_batch).sum().item()
        else:
            break
    
    return correct_predictions / num_examples

# Computes the loss for a single batch obtained from the previously defined data loaders
def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)[:, -1, :]  # 1
    loss = torch.nn.functional.cross_entropy(logits, target_batch)
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.0
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:  # 1
        num_batches = min(num_batches, len(data_loader))

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break

    return total_loss / num_batches

def train_classifier_simple(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    num_epochs,
    eval_freq,
    eval_iter
):
    """Train a classifier model with validation tracking."""
    
    # Initialize tracking lists
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    examples_seen, global_step = 0, -1
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        
        for input_batch, target_batch in train_loader:
            # Zero gradients from previous step
            optimizer.zero_grad()
            
            # Calculate loss for current batch
            loss = calc_loss_batch(
                input_batch, target_batch, model, device
            )
            
            # Backpropagation
            loss.backward()
            
            # Update model parameters
            optimizer.step()
            
            # Track number of examples processed
            examples_seen += input_batch.shape[0]
            # Increment step counter
            global_step += 1
            
            # Periodic evaluation
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print(
                    f"Ep {epoch+1} (Step {global_step:06d}): "
                    f"Train loss {train_loss:.3f}, "
                    f"Val loss {val_loss:.3f}"
                )
        
        # Calculate epoch-end accuracies
        train_accuracy = calc_accuracy_loader(
            train_loader, model, device, num_batches=eval_iter
        )
        val_accuracy = calc_accuracy_loader(
            val_loader, model, device, num_batches=eval_iter
        )
        
        # Display accuracy metrics
        print(f"Training accuracy: {train_accuracy*100:.2f}% | ", end="")
        print(f"Validation accuracy: {val_accuracy*100:.2f}%")
        
        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)
    
    return train_losses, val_losses, train_accs, val_accs, examples_seen

def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    """Evaluate model on training and validation sets."""
    
    model.eval()
    
    with torch.no_grad():
        train_loss = calc_loss_loader(
            train_loader, model, device, num_batches=eval_iter
        )
        val_loss = calc_loss_loader(
            val_loader, model, device, num_batches=eval_iter
        )
    
    model.train()
    
    return train_loss, val_loss

def classify_review(
    text,
    model,
    tokenizer,
    device,
    max_length=None,
    pad_token_id=50256
):
    """Classify a text review as spam or not spam."""
    
    model.eval()
    
    # Tokenize input text
    input_ids = tokenizer.encode(text)
    
    # Truncate to supported context length
    supported_context_length = model.pos_emb.weight.shape[0]
    input_ids = input_ids[:min(
        max_length, supported_context_length
    )]
    
    # Pad sequence to max_length
    input_ids += [pad_token_id] * (max_length - len(input_ids))
    
    # Convert to tensor and add batch dimension
    input_tensor = torch.tensor(
        input_ids, device=device
    ).unsqueeze(0)
    
    # Get model predictions without gradient computation
    with torch.no_grad():
        logits = model(input_tensor)[:, -1, :]
    
    # Get predicted class label
    predicted_label = torch.argmax(logits, dim=-1).item()
    
    return "spam" if predicted_label == 1 else "not spam"