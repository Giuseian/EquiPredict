from utility_functions import * 
from options import * 
from model.equipredict import EquiPredict
from train_test_val import * 

device = torch.device("cuda" if args.cuda else "cpu")


# Final results function
def final_results():
    # Seed setup
    if args.seed >= 0:
        seed = args.seed
        setup_seed(seed)
    else:
        seed = random.randint(0, 1000)
        setup_seed(seed)
    print('The seed is:', seed)

    # Model setup
    model = EquiPredict(
        node_features=args.past_length, 
        edge_features=2, 
        hidden_dim=args.nf, 
        input_dim=args.past_length, 
        hidden_channel_dim=args.channels, 
        output_dim=args.future_length, 
        device=device, 
        act_fn=nn.SiLU(), 
        layers=args.n_layers, 
        coords_weight=1.0, 
        use_recurrent=False, 
        normalize_diff=False, 
        use_tanh=args.tanh
    )

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    results = {'epochs': [], 'train_losses': [], 'val_losses': [], 'train_preds': [], 'train_gt': []}
    best_val_loss = 1e8
    best_val_ade = 1e8
    best_epoch = 0
    lr_now = args.lr

    for epoch in range(0, args.epochs):
        # Apply learning rate decay if specified
        if args.apply_decay:
            if epoch % args.epoch_decay == 0 and epoch > 0:
                lr_now = lr_decay(optimizer, lr_now, args.lr_gamma)
        
        # Train the model
        train_loss, train_preds, train_gt = train(model, optimizer, epoch, loader_train, device)
        results['train_losses'].append(train_loss)
        results['train_preds'].append(train_preds)
        results['train_gt'].append(train_gt)
        print(f'Epoch {epoch}: Train Loss: {train_loss:.5f}')
        
        # Evaluate on validation set
        val_loss, val_ade = validate(model, optimizer, epoch, loader_val, device)
        results['val_losses'].append(val_loss)
        print(f'Epoch {epoch}: Validation Loss: {val_loss:.5f}, Validation ADE: {val_ade:.5f}')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_ade = val_ade
            best_epoch = epoch
            state = {'epoch': epoch,
                     'state_dict': model.state_dict(),
                     'optimizer': optimizer.state_dict()}
            file_path = os.path.join(args.model_save_dir, f'{args.subset}_ckpt_best.pth')
            torch.save(state, file_path)

        clear_cache()

        # Save intermediate results to reduce memory usage
        if epoch % 5 == 0:  # Adjust frequency as needed
            results_path = os.path.join('/kaggle/working/saved_models', f'training_results_epoch_{epoch}.pkl')
            with open(results_path, 'wb') as f:
                pickle.dump({'results': results}, f)
            clear_cache()

    # Save the final model at the end of the training and validation phases 
    state = {'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
    file_path = os.path.join(args.model_save_dir, f'{args.subset}_final.pth')
    torch.save(state, file_path)
    
    # Load the trained model
    model_trained_path = file_path 
    print('Loading model from:', model_trained_path)
    model_ckpt = torch.load(model_trained_path)
    model.load_state_dict(model_ckpt['state_dict'], strict=False)
    test_loss, ade = test(model, loader_test, device)
    print('ADE final:', ade, 'FDE final:', test_loss)
    
    # Save the final results dictionary
    results_path = os.path.join('/kaggle/working/saved_models', 'training_results.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump({'results': results}, f)

    clear_cache()
    return results

results = final_results() 