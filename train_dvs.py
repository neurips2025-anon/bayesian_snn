import torch

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

import numpy as np
import itertools
import time, random, argparse
import math

import torchvision.transforms as transforms

from dvs_data import DVSGestureChain

def resize_to_32(frames):
    """
    Converts incoming event frames/arrays to a float32 tensor and
    bilinearly resizes each frame to 32×32.
    """
    frames = torch.as_tensor(frames, dtype=torch.float32)
    return F.interpolate(frames, size=(32, 32),
                         mode='bilinear', align_corners=False)

def normal_pdf(x):
    return torch.exp(-0.5 * x**2) / math.sqrt(2 * math.pi)

def normal_cdf(x):
    return 0.5 * (1 + torch.erf(x / math.sqrt(2)))

def heaviside(x):
    return (x > 0).float()


class Activation_SG(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, noise):
        output = heaviside(input + noise)
        ctx.save_for_backward(input)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors

        grad_input = (1/0.4)*normal_pdf(input/0.4) * grad_output

        return grad_input, None


@torch.jit.script
def activation_fwd(input, noise):
    return (input + noise > 0).float()

@torch.jit.script
def activation_bwd(input, combined_sigscale, grad_output, output):
    safe_scale = torch.clamp(combined_sigscale, min=1e-6)
    scaled_input = input / safe_scale

    pdf_val = (combined_sigscale > 0).float() *torch.exp(-0.5 * scaled_input * scaled_input) / (math.sqrt(2 * math.pi)*safe_scale)

    grad_input =  (pdf_val) * grad_output 
    grad_scale =- (input * grad_input) / safe_scale

    return grad_input, grad_scale

    
class Activation_Bayes(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, noise, combined_sigscale):
        output = activation_fwd(input, noise)
        ctx.save_for_backward(input, combined_sigscale, output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, combined_sigscale, output = ctx.saved_tensors
        grad_input, grad_scale = activation_bwd(input, combined_sigscale, grad_output,output)
        return grad_input, None, grad_scale



# ---- DataLoader ----


def get_dataloaders(batch_size):

    dataset_dir='./'

    transform = transforms.Compose([
    transforms.Lambda(resize_to_32),   # <-- named function
])


    train_dataset = DVSGestureChain(root=dataset_dir, split='train', frames_number=time_steps, seq_len=1,class_num=11, transform=transform)
    test_dataset =DVSGestureChain(root=dataset_dir, split='test', frames_number=time_steps, seq_len=1,class_num=11, transform=transform)
    val_dataset =DVSGestureChain(root=dataset_dir, split='validation', frames_number=time_steps, seq_len=1,class_num=11, transform=transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True, pin_memory=True, num_workers=4, persistent_workers=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=128, #512
        shuffle=False, pin_memory=True,num_workers=4, persistent_workers=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=128, #512
        shuffle=False, pin_memory=True,num_workers=4, persistent_workers=True
    )


    return train_loader, test_loader, val_loader

# ---- Custom Network Definition ----

class CustomNet(nn.Module):
    def __init__(self, input_channels, conv_params, output_size):
        super(CustomNet, self).__init__()
        self.output_size = output_size
        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
     
        self.scalers = nn.ParameterList()
        self.biases = nn.ParameterList()

        self.scales = nn.ParameterList() if scale_style == "learnable" else []
        self.scales_biases = nn.ParameterList() if scale_style == "learnable" else []
        self.thresholds =  []
        self.tau_mem_transform =  []

        in_channels = input_channels

        for i, params in enumerate(conv_params):
            out_channels, kernel_size, stride, padding = params

            for t in range(time_steps):
                self.bn_layers.append(nn.BatchNorm2d(num_features=out_channels))

                self.scalers.append(nn.Parameter(1*torch.ones((1,out_channels,1,1),device=device)))
                self.biases.append(nn.Parameter(0*torch.ones((1,out_channels,1,1),device=device)))
        
        
            conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
            fanin = torch.tensor(conv_layer.weight.shape[1] *
                                    conv_layer.weight.shape[2] *
                                    conv_layer.weight.shape[3], dtype=torch.float32)

            nn.init.uniform_(conv_layer.weight, -1/np.sqrt(fanin), 1/np.sqrt(fanin))

            self.conv_layers.append(conv_layer)

            if scale_style == "learnable":
                scale_shape = (1, 1, 1)
                self.scales.append(nn.Parameter(torch.log(torch.exp((0.5)/(fanin**0.5)) - 1).repeat(scale_shape)))

                self.scales_biases.append(nn.Parameter(-1*torch.ones(scale_shape)))

            else:
                scale_shape = (1, 1, 1, 1)
                self.scales.append(torch.tensor(torch.log(torch.exp((0.5)/(fanin**0.5)) - 1).repeat(scale_shape), device=device))

                self.scales_biases.append(torch.tensor(-1*torch.ones(scale_shape)))

    
            self.thresholds.append(torch.ones((out_channels, 1, 1), device=device) * 1)
            self.tau_mem_transform.append(torch.ones((out_channels, 1, 1), device=device) * 10)

            in_channels = out_channels

        self.fc2 = nn.Linear(self.conv_layers[-1].weight.shape[0], output_size)
        fanin = self.fc2.weight.shape[1]
        nn.init.uniform_(self.fc2.weight, -1/np.sqrt(fanin), 1/np.sqrt(fanin))

        self.tau_mem_out_transform = torch.ones(self.fc2.weight.shape[0], device=device) * 20


        self.apool = nn.AvgPool2d(2, 2)

        self.gpool = nn.AdaptiveAvgPool2d(1)


        self.mem_poten_noiseless = None
        self.total_var = None
        self.total_sd = None
        self.spike_list = None
        self.output_mem_poten = None

    def allocate_internal_states(self, b_size, device):
        mem_shapes = []
        for i, params in enumerate(conv_params):
            factor = 1 + 1*(i > (num_blocks * 2)) + 2 * (i > 2 * (num_blocks * 2))#+ 4 * (i > 3 * (num_blocks * 2))
            mem_shapes.append((b_size, params[0], 32 // factor, 32 // factor))

        self.mem_poten_noiseless = []
        self.total_var = []
        self.spike_list = []

        for shape in mem_shapes:
            self.mem_poten_noiseless.append(torch.zeros(shape, device=device))
            self.total_var.append(torch.zeros(shape, device=device))
        
            self.spike_list.append(torch.zeros(shape, device=device))

        self.output_mem_poten = torch.zeros(b_size, self.output_size, device=device)

    def forward(self, data_in, count=1, noise_style="noisy"):
        device = data_in.device
        b_size = data_in.size(0)
        self.allocate_internal_states(b_size, device)

        tau_mem = []
        beta_mem = []
        for layer in range(len(self.tau_mem_transform)):
            tau_mem.append(F.softplus(self.tau_mem_transform[layer]))
            beta_mem.append(torch.exp(-delta_t / tau_mem[layer]))


        reg_mems = []
        reg_combined_sigscale = []
        spike_list_reg=[]

        for t in range(time_steps):
            data=data_in[:,t,:,:,:]
            identity=0
            for j in range(len(self.conv_layers)):
                if model_style == "bayesian":
                    if j % (num_blocks * 2) == 1 and j != 1:
                        ds_spikes=self.apool(self.spike_list[j-1])
                        identity=torch.cat([ds_spikes, torch.zeros_like(ds_spikes)], dim=1)
                  
                    elif j % 2 == 1:
                        identity=self.spike_list[j - 1].clone()

                    e_sigscale = F.softplus(self.scales[j])
                    if torch.isnan(e_sigscale).any():
                        raise ValueError("NaN detected!")

                    e_sigscale_bias=F.softplus(self.scales_biases[j])

                    if j == 0:
                        weight =torch.ones((1, data.shape[1], self.conv_layers[j].kernel_size[0], self.conv_layers[j].kernel_size[0]),device=device)#*getattr(model,f"conv_mask_{j}")  # Fixed to 1
                        added_var_in=(e_sigscale**2)*F.conv2d(data**2, weight, bias=None, stride=self.conv_layers[j].stride, padding=1)

                    else:
                        weight =torch.ones((1, self.spike_list[j - 1].shape[1], self.conv_layers[j].kernel_size[0], self.conv_layers[j].kernel_size[0]),device=device)#*getattr(model,f"conv_mask_{j}")  # Fixed to 1
                        added_var_in=(e_sigscale**2)*F.conv2d(self.spike_list[j - 1].detach()**2, weight, bias=None, stride=self.conv_layers[j].stride, padding=1)


                    added_var = added_var_in

                    self.total_var[j] =(beta_mem[j]**2) * self.total_var[j] + added_var 
                    
                    total_var_mask = ((self.total_var[j]+e_sigscale_bias**2) > 0).float()
                    total_var_safe = torch.clamp(self.total_var[j]+e_sigscale_bias**2, min=1e-6)
               
                    self.total_sd = total_var_mask * torch.sqrt(total_var_safe)

                    added_noise = torch.tensor(0)
                    if noise_style == "noisy":
                        added_noise = (torch.normal(0, 1, size=self.total_sd.shape, device=device) * self.total_sd).detach()

                    if j == 0:
                        new_input=self.conv_layers[0](data)*self.scalers[time_steps * j + t]+self.biases[time_steps * j + t]

                        self.mem_poten_noiseless[j] = beta_mem[j] * self.mem_poten_noiseless[0] + new_input

                    else:
                        new_input=self.conv_layers[j](self.spike_list[j - 1])*self.scalers[time_steps * j + t]+self.biases[time_steps * j + t]

                        self.mem_poten_noiseless[j] = beta_mem[j] * self.mem_poten_noiseless[j] + new_input

                    
                    reg_mems.append(self.mem_poten_noiseless[j])
                    reg_combined_sigscale.append(self.total_sd)


                    heaviside_input = self.mem_poten_noiseless[j] - self.thresholds[j]
                    self.spike_list[j] = Activation_Bayes.apply(heaviside_input, added_noise, self.total_sd)

                    self.mem_poten_noiseless[j]=self.mem_poten_noiseless[j]- self.thresholds[j] * self.spike_list[j]

                    if j % 2 == 0 and j != 0:
                        self.spike_list[j] = self.spike_list[j]+ identity

                  

                elif model_style == "sg":

                    if j % (num_blocks * 2) == 1 and j != 1:
                      
                        ds_spikes=self.apool(self.spike_list[j-1])
                        identity=torch.cat([ds_spikes, torch.zeros_like(ds_spikes)], dim=1)
                    
                    elif j % 2 == 1:
                        identity = self.spike_list[j - 1].clone()

                    if j == 0:
                        self.mem_poten_noiseless[0] = beta_mem[0] * self.mem_poten_noiseless[0] +self.conv_layers[0](data)*self.scalers[time_steps * j + t]+self.biases[time_steps * j + t] #+self.bn_layers[time_steps * j + t](self.conv_layers[0](data)) 

                    else:
                        self.mem_poten_noiseless[j] = beta_mem[j] * self.mem_poten_noiseless[j] +self.conv_layers[j](self.spike_list[j - 1])*self.scalers[time_steps * j + t]+self.biases[time_steps * j + t] #+self.bn_layers[time_steps * j + t](self.conv_layers[j](self.spike_list[j - 1])) 


                    total_noise = 0
                    heaviside_input = self.mem_poten_noiseless[j] - self.thresholds[j]


                    self.spike_list[j] = Activation_SG.apply(heaviside_input, total_noise)


                    self.mem_poten_noiseless[j]=self.mem_poten_noiseless[j]- self.thresholds[j] * self.spike_list[j]


                    if j % 2 == 0 and j != 0:
                        self.spike_list[j] = self.spike_list[j] + identity

            

            x = self.gpool(self.spike_list[-1])
            x = x.view(x.shape[0], -1)
            self.output_mem_poten = self.output_mem_poten + self.fc2(x)

        output = self.output_mem_poten / time_steps
        return output, reg_mems, reg_combined_sigscale

# ---- Evaluation Function ----

def evaluate(model, test_loader, val_loader,eval_device):
    model.eval()
    correct = 0
    total = 0
    test_loss = 0.0

    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(eval_device) #, memory_format=torch.channels_last)
            target = target.to(eval_device)

           # with autocast():
            output, _, _= model(data)
            loss = F.cross_entropy(output, target, reduction='mean')
            test_loss += loss.item() * data.size(0)
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
        test_accuracy = correct / total
        test_avg_loss = test_loss / total

        correct = 0
        total = 0
        test_loss = 0.0
        for data, target in val_loader:
            data = data.to(eval_device) 
            target = target.to(eval_device)

            output, _, _= model(data)
            loss = F.cross_entropy(output, target, reduction='mean')
            test_loss += loss.item() * data.size(0)
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
        val_accuracy = correct / total

    return test_accuracy, test_avg_loss, val_accuracy

# ---- Training Function ----
def adjust_learning_rate(optimiser, epoch, total_epochs):
   

    lr1_init = 0.005  
    lr2_init = 0.05
    min_lr1 = lr1_init / 50    
    min_lr2 = lr2_init /50


    lr1 = min_lr1 + (lr1_init - min_lr1) * (1 + math.cos(math.pi * (epoch) / total_epochs)) / 2
    lr2 = min_lr2 + (lr2_init - min_lr2) * (1 + math.cos(math.pi * (epoch) / total_epochs)) / 2

    
    optimiser.param_groups[0]['lr']=lr1
    optimiser.param_groups[1]['lr']=lr1
    optimiser.param_groups[2]['lr']=lr2 
    optimiser.param_groups[3]['lr']=lr1
    optimiser.param_groups[4]['lr']=lr1 
    optimiser.param_groups[5]['lr']=lr1
    optimiser.param_groups[6]['lr']=lr2
        

def train_model(model, train_loader, epochs, device):

    param_groups = [
        {'params': itertools.chain(*[layer.parameters() for layer in model.conv_layers]), 'lr': 0.005},
        {'params': model.fc2.parameters(), 'lr': 0.005},
        {'params': model.scales, 'lr': 0.005},
        {'params': itertools.chain(*[layer.parameters() for layer in model.bn_layers]), 'lr': 0.005},
        {'params': model.scalers, 'lr': 0.005},
        {'params': model.biases, 'lr': 0.005},
        {'params': model.scales_biases, 'lr': 0.005},
        ]

    optimiser = optim.Adam(param_groups)

   
    train_losses = []
    train_accuracies = []
    test_losses = []
    accuracies = []

    for epoch in range(epochs):

        adjust_learning_rate(optimiser, epoch, epochs)

        model.train()
        epoch_loss = 0.0
        total = 0
        correct = 0

        count = 0

        for data, target in train_loader:
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            optimiser.zero_grad()

        
            output, reg_mems, reg_combined_sigscale = model(data, count)
            primary_loss = F.cross_entropy(output, target, reduction='mean')
        
            reg_loss = 0
            if len(reg_mems)>0:
                for i in range(len(reg_mems)):
                    mask = reg_combined_sigscale[i] > 0
                    safe_sigscale = torch.clamp(reg_combined_sigscale[i], min=1e-6)
                    ratio = mask*(reg_mems[i] / safe_sigscale)

                    reg_loss += torch.mean(torch.sum(ratio**2, dim=(1,2,3)))
                

            lambda_reg =1e-10

            loss = primary_loss + lambda_reg * reg_loss


            loss.backward()
            optimiser.step()


            epoch_loss += loss.item() * data.size(0)
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            count += 1

            if count % 50 == 0:
                print(f"Epoch {epoch+1}, Batch {count}")

            

        avg_train_loss = epoch_loss / total
        avg_train_accuracy = correct / total
        train_losses.append(avg_train_loss)
        train_accuracies.append(avg_train_accuracy)

        test_accuracy, avg_test_loss, val_accuracy = evaluate(model, test_loader, val_loader,device)
        test_losses.append(avg_test_loss)
        accuracies.append(test_accuracy)


        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f} - Train Acc: {avg_train_accuracy:.4f}  - Val Acc: {val_accuracy:.4f}  - Test Loss: {avg_test_loss:.4f} - Test Acc: {test_accuracy:.4f}")

    return train_losses, test_losses, accuracies

# ---- Global Settings ----

input_channels = 2
output_size = 11 
num_blocks = 3 

conv_params = [
    (32, 3, 1, 1),
    (32, 3, 1, 1),
    (32, 3, 1, 1),
    (32, 3, 1, 1),
    (32, 3, 1, 1),
     (32, 3, 1, 1),
    (32, 3, 1, 1),
    
    (64, 3, 2, 1),
    (64, 3, 1, 1),
     (64, 3, 1, 1),
    (64, 3, 1, 1),
    (64, 3, 1, 1),
    (64, 3, 1, 1),

    (128, 3, 2, 1),
    (128, 3, 1, 1),
    (128, 3, 1, 1),
    (128, 3, 1, 1),
     (128, 3, 1, 1),
    (128, 3, 1, 1),
 ]

delta_t = 1


results = {}

# ---- Main ----

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default="bayesian")
    parser.add_argument("--scale_style", type=str, default="learnable")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs",     type=int, default=70)
    parser.add_argument("--time_steps",     type=int, default=49)
    parser.add_argument("--seed",       type=int, default=3)
    args = parser.parse_args()

    scale_style=args.scale_style
    model_style=args.method
    time_steps = args.time_steps

    # set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader, val_loader = get_dataloaders(batch_size=args.batch_size)


    method =  f"{args.method},{args.scale_style}"


    print(f"\nTraining method: {method}")
    start_time = time.time()

    model = CustomNet(
        input_channels=input_channels,
        conv_params=conv_params,
        output_size=output_size
    ).to(device)

    train_losses, test_losses, accuracies = train_model(model, train_loader, epochs=args.epochs,device=device)

    results[method] = {
        'train_loss': train_losses,
        'test_loss': test_losses,
        'accuracy': accuracies,
      
    }

    end_time = time.time()
    print(f"\nTotal training time for {method}: {end_time - start_time:.2f} seconds")

    fname = f"{model_style}_dvs.pth"

    checkpoint = {
    'model_state_dict': model.state_dict(),
    'results': results,
    }

    torch.save(checkpoint, fname)

    
