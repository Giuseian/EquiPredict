from utility_functions import * 

def train(model, optimizer, epoch, loader, device, backprop=True):
    all_predictions = []
    all_gt = []
    
    start_time = time.time()
    model.train() if backprop else model.eval()
    res = {'epoch': epoch, 'loss': 0, 'counter': 0}
    
    for batch_idx, data in enumerate(loader):
        if data is not None:
            pre_data, fut_data, num_valid = data
            pre_data, fut_data, num_valid = pre_data.to(device), fut_data.to(device), num_valid.to(device).type(torch.int)
            
            vel = torch.zeros_like(pre_data).to(device)
            vel[:, :, 1:] = pre_data[:, :, 1:] - pre_data[:, :, :-1]
            vel[:, :, 0] = vel[:, :, 1]
            
            batch_size, agent_num, length, _ = pre_data.size()
            optimizer.zero_grad()
            
            nodes = torch.sqrt(torch.sum(vel ** 2, dim=-1)).detach()
            loc_pred, category = model(nodes, pre_data.detach(), vel, num_valid, agent_num)
            fut_data = fut_data[:, :, None, :, :]
            
            if args.supervise_all:
                mask = get_valid_mask2(num_valid, pre_data.size(1)).to(device)[:, :, None, :, :]
                loss = torch.mean(torch.min(torch.mean(torch.norm(mask * (loc_pred - fut_data), dim=-1), dim=3), dim=2)[0])
                # if args.supervise_all, the loss is the mean of the minimum norms of the difference between predicted and true locations
            else:
                loss = torch.mean(torch.min(torch.mean(torch.norm(loc_pred[:, 0:1] - fut_data[:, 0:1], dim=-1), dim=-1), dim=-1)[0])
                # if not args.supervise_all, the loss is computed only for the first agent (ego agent)
            
            if backprop:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)  # Clip gradients to avoid explosion
                optimizer.step()

            res['loss'] += loss.item() * batch_size
            res['counter'] += batch_size
            
            all_predictions.append(loc_pred.detach().cpu())
            all_gt.append(fut_data.detach().cpu())
    
    avg_loss = res['loss'] / res['counter']
    epoch_time = time.time() - start_time
    
    all_predictions = torch.cat(all_predictions, dim=0)
    all_gt = torch.cat(all_gt, dim=0)

    print(f"{'==> ' if not backprop else ''}epoch {epoch} avg train loss: {avg_loss:.5f}, time taken: {epoch_time:.2f} seconds")
    return avg_loss, all_predictions, all_gt


def validate(model, optimizer, epoch, loader, device):
    start_time = time.time()
    model.eval()
    res = {'epoch': epoch, 'loss': 0, 'counter': 0, 'ade': 0}
    
    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            if data is not None:
                pre_data, fut_data, num_valid = data
                pre_data, fut_data, num_valid = pre_data.to(device), fut_data.to(device), num_valid.to(device).type(torch.int)

                vel = torch.zeros_like(pre_data).to(device)
                vel[:, :, 1:] = pre_data[:, :, 1:] - pre_data[:, :, :-1]
                vel[:, :, 0] = vel[:, :, 1]
                
                batch_size, agent_num, length, _ = pre_data.size()
                optimizer.zero_grad()
                
                nodes = torch.sqrt(torch.sum(vel ** 2, dim=-1)).detach()
                loc_pred, category_list = model(nodes, pre_data.detach(), vel, num_valid, agent_num)


                loc_pred = loc_pred.cpu().numpy()
                fut_data = fut_data.cpu().numpy()[:, :, None, :, :]
                ade = np.mean(np.min(np.mean(np.linalg.norm(loc_pred[:, 0:1] - fut_data[:, 0:1], axis=-1), axis=-1), axis=-1))
                # ade measures the average distance between predicted and ground truth locations over all time steps. It is computed as the mean of the minimum error across time steps 
                fde = np.mean(np.min(np.mean(np.linalg.norm(loc_pred[:, 0:1, :, -1:] - fut_data[:, 0:1, :, -1:], axis=-1), axis=-1), axis=-1))
                # fde measures the average distance between predicted and ground truth locations over all time steps. It is computed as the mean of the minimum error across time steps
                
                res['loss'] += fde*batch_size
                res['ade'] += ade*batch_size
                res['counter'] += batch_size
                
    res['ade'] *= args.test_scale
    res['loss'] *= args.test_scale
    epoch_time = time.time() - start_time
    print(f"==> epoch {epoch} avg val loss: {res['loss'] / res['counter']:.5f} ade: {res['ade'] / res['counter']:.5f}, time taken: {epoch_time:.2f} seconds")
    
    return res['loss'] / res['counter'], res['ade'] / res['counter'] 


def test(model, loader, device):
    start_time = time.time()
    model.eval()
    res = {'loss': 0, 'counter': 0, 'ade': 0}

    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            if data is not None:
                pre_data, fut_data, num_valid = data
                pre_data, fut_data, num_valid = pre_data.to(device), fut_data.to(device), num_valid.to(device).type(torch.int)

                vel = torch.zeros_like(pre_data).to(device)
                vel[:, :, 1:] = pre_data[:, :, 1:] - pre_data[:, :, :-1]
                vel[:, :, 0] = vel[:, :, 1]
                
                batch_size, agent_num, length, _ = pre_data.size()
                #optimizer.zero_grad()
                
                nodes = torch.sqrt(torch.sum(vel ** 2, dim=-1)).detach()
                loc_pred, category_list = model(nodes, pre_data.detach(), vel, num_valid, agent_num)
                
                loc_pred = loc_pred.cpu().numpy()
                fut_data = fut_data.cpu().numpy()[:, :, None, :, :]
                ade = np.mean(np.min(np.mean(np.linalg.norm(loc_pred[:, 0:1] - fut_data[:, 0:1], axis=-1), axis=-1), axis=-1))
                # ade measures the average distance between predicted and ground truth locations over all time steps. It is computed as the mean of the minimum error across time steps
                fde = np.mean(np.min(np.mean(np.linalg.norm(loc_pred[:, 0:1, :, -1:] - fut_data[:, 0:1, :, -1:], axis=-1), axis=-1), axis=-1))
                # fde measures the average distance between predicted and ground truth locations over all time steps. It is computed as the mean of the minimum error across time steps
                
                res['loss'] += fde*batch_size
                res['ade'] += ade*batch_size
                res['counter'] += batch_size
                
    res['ade'] *= args.test_scale
    res['loss'] *= args.test_scale
    epoch_time = time.time() - start_time
    print(f"Test avg loss: {res['loss'] / res['counter']:.5f} ade: {res['ade'] / res['counter']:.5f}, time taken: {epoch_time:.2f} seconds")

    
    return  res['loss'] / res['counter'], res['ade'] / res['counter']