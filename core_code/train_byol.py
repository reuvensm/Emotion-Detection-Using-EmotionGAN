from lightly.models.utils import update_momentum
from core_code.core_utils.model_utils import save_model
from core_code.core_utils.logger_config import logger
import time


def train_byol(epochs, criterion, optimizer, device, dataloader, model, net_name, exp_name):
    model.train()
    for epoch in range(epochs):
        epoch_time = time.time()
        total_loss = 0
        for data in dataloader:
            x1, _, x0  = data # x0 is the original, x1 is the augmented
            update_momentum(model.backbone, model.backbone_momentum, m=0.99)
            update_momentum(model.projection_head, model.projection_head_momentum, m=0.99)
            x0 = x0.to(device)
            x1 = x1.to(device)
            p0 = model(x0)
            z0 = model.forward_momentum(x0)
            p1 = model(x1)
            z1 = model.forward_momentum(x1)
            loss = 0.5 * (criterion(p0, z1) + criterion(p1, z0))
            total_loss += loss.detach()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        log_line = "Epoch: {} | Loss: {:.4f} | Epoch Time: {:.2f} secs".format(epoch, total_loss, time.time()-epoch_time)
        print(log_line)
        logger.info(log_line)
    save_model(model, optimizer, total_loss, epoch, f'ckpt_byol_{net_name}_{exp_name}_last_trained_epoch.pth', byol=True)  # Save in case of failure
