import os, torch
class CheckpointManager:
    """
    checkpoint standard:
    {
        'global_step': int,
        'global_epoch': int,
        'block_size':int,
        'vocab_size':int,
        'n_layer':int,
        'n_head':int,
        'n_embd':int,
        'bias':bool,
        'dataset': str,
        'state_dict': dict,
        'optimizer': dict,
    }
    """
    def __init__(self, save_root):
        self.save_root = save_root
        if not os.path.exists(save_root):
            os.makedirs(save_root)
    def load(self, path, model, optimizer=None):
        print(f'Loading State Dict from {path}')
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['state_dict'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
        global_step = checkpoint['global_step']
        global_epoch = checkpoint['global_epoch']
        print('state dict config:')
        for k, v in checkpoint.items():
            if k == 'state_dict' or k == 'optimizer':
                continue
            print(f'{k}={v}')
        print('-'*6)
        return model, global_step, global_epoch

    def save(self, model, config, global_step, global_epoch, optimizer):
        filename = f'cpt{global_step}.pth'
        path = os.path.join(self.save_root, filename)
        checkpoint = {
            'global_step': global_step,
            'global_epoch': global_epoch,
            'block_size': config.block_size,
            'vocab_size': config.vocab_size,
            'n_layer': config.n_layer,
            'n_head': config.n_head,
            'n_embd': config.n_embd,
            'bias': config.bias,
            'dataset': config.data_path,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(checkpoint, path)