from config import *
from unet import UNet
from tiloss import Finalloss
from glob import glob
from numba import jit
import warnings
from dataloader import *
import neptune 

warnings.filterwarnings("ignore")
final_loss = Finalloss()
unet = UNet(3, 3)

run = neptune.init_run(
    project="aureliensihab/deep-icy",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIwZjdkOTI0Yy1iOGJkLTQyMzEtYmEyOC05MmFmYmFhMWExNTMifQ==",
)  # your credentials

params = {"learning_rate":parameters["lr"],
          "optimizer": "Adam",
          "ti_loss_val": 1e-1}

run["parameters"] = params

optimizer = torch.optim.Adam(unet.parameters(), lr=parameters["lr"])
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)


def train(model, optimizer, train_dl, val_dl, epochs=100):
    tmp = (torch.ones(1) * 1e15).cuda()
    for epoch in range(1, epochs + 1):
        # --- TRAIN AND EVALUATE ON TRAINING SET -----------------------------
        model.train()
        model.cuda()
        loss_tot = 0.0
        num_train_correct = 0
        num_train_examples = 0

        for batch in train_dl:
            optimizer.zero_grad()

            images = batch[0].cuda()
            outputs = batch[1].cuda()
            pred_outputs = model(images)

            loss = final_loss(pred_outputs, outputs)
            run['train/epoch/loss'].log(loss)
            loss_tot = loss
            loss_tot.backward()
            optimizer.step()

        # --- EVALUATE ON VALIDATION SET -------------------------------------
        model.eval()
        val_loss_tot = 0.0
        num_val_correct = 0
        num_val_examples = 0

        mean = torch.zeros(1).cuda()
        with torch.no_grad():
            for batch in val_dl:
                optimizer.zero_grad()
                images = batch[0].cuda()
                outputs = batch[1].cuda()

                pred_outputs = model(images)

                loss = final_loss(pred_outputs, outputs)
                val_loss_tot = loss
                run['validation/epoch/loss'].log(val_loss_tot)
                mean += val_loss_tot
                optimizer.step()
            mean = torch.mean(mean)

            if torch.gt(tmp, mean):
                print("the val loss decreased: saving the model...")
                tmp = mean
                path_weights = "weights"
                torch.save(model.state_dict(), path_weights)
    return "Training done: the model was trained for " + str(epochs) + " epochs"


train(unet,
      optimizer,
      dataloaders["train"],
      dataloaders["val"],
      epochs=100)
