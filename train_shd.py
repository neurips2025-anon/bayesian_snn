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
from torchvision.datasets import CIFAR10

from shd_data import get_shd_datasets


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


    train_dataset, test_dataset = get_shd_datasets(
            cache_dir="./",  
            cache_subdir="hdspikes",   
            time_steps=time_steps,
            max_time=1.4,
            input_size=700,
            force_preprocess=True
        )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True, pin_memory=True, num_workers=4, persistent_workers=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=512,
        shuffle=False, pin_memory=True,num_workers=4, persistent_workers=True
    )


    return train_loader, test_loader

# ---- Custom Network Definition ----

class CustomNet(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
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

        self.weights = nn.ParameterList()

        for i in range(len(hidden_sizes)):
            in_size = input_size if i == 0 else hidden_sizes[i-1]
            out_size = hidden_sizes[i]
            weight = nn.Parameter(torch.empty(in_size, out_size))
  
            weight=nn.init.uniform(weight, -1/np.sqrt(in_size),1/np.sqrt(in_size))

            self.weights.append(weight)

            for t in range(time_steps):
                self.bn_layers.append(nn.BatchNorm1d(num_features=hidden_sizes[i]))

                self.scalers.append(nn.Parameter(1*torch.ones((hidden_sizes[i]),device=device)))
                self.biases.append(nn.Parameter(0*torch.ones((hidden_sizes[i]),device=device)))


        self.output_weights = nn.Parameter(torch.empty(hidden_sizes[-1], output_size))   
  
        nn.init.uniform_(self.output_weights, -1/np.sqrt(hidden_sizes[-1]),1/np.sqrt(hidden_sizes[-1]))

        self.recurrent_weights = nn.ParameterList()
        self.masks=[]
        
        for i in range(len(hidden_sizes)):
            size = hidden_sizes[i]
            recurrent_weight = nn.Parameter(torch.empty(size, size))
      
            recurrent_weight=nn.init.uniform(recurrent_weight, -1/np.sqrt(size),1/np.sqrt(size))
       
            self.recurrent_weights.append(recurrent_weight)

            self.masks.append(torch.ones(size,size,device=device)-torch.eye(size,device=device))

    
        if scale_style=="learnable":
            self.scales = nn.ParameterList() 
            self.scales_rec = nn.ParameterList()
            for i in range(len(hidden_sizes)):
                scale_shape=1
                if i==0:
                    fanin=torch.tensor(input_size)
                    self.scales.append(nn.Parameter((torch.log(torch.exp((0.5*1)/(fanin**0.5))-1).repeat(scale_shape)))) 

                     
                else:
                    fanin=torch.tensor(hidden_sizes[i-1])
                    self.scales.append(nn.Parameter((torch.log(torch.exp((0.5*1)/(fanin**0.5))-1).repeat(scale_shape)))) 
                 

                fanin=torch.tensor(hidden_sizes[i])
                self.scales_rec.append(nn.Parameter((torch.log(torch.exp((0.5*1)/(fanin**0.5))-1).repeat(scale_shape))))       

                self.scales_biases.append(nn.Parameter(-2*torch.ones(scale_shape)))   


        elif scale_style=="fixed":
            self.scales=[]
            self.scales_rec =[torch.ones(1,requires_grad=False,device=device)]
            for i in range(len(hidden_sizes)):
                if i==0:
                    self.scales.append(torch.ones(1,device=device)*0)
                else:
                    self.scales.append(torch.ones(1,device=device)*0)

                self.scales_biases.append((-1*torch.ones(1,device=device))) 


      
        self.thresholds= [torch.ones(hidden_sizes[i],device=device)*1 for i in range(len(hidden_sizes))]
        self.tau_mem_transform = [torch.ones(hidden_sizes[i],device=device)*10 for i in range(len(hidden_sizes))] #10

        self.apool = nn.AvgPool2d(2, 2)

        self.gpool = nn.AdaptiveAvgPool2d(1)


        self.mem_poten_noiseless = None
        self.total_var = None
        self.total_sd = None
        self.spike_list = None
        self.output_mem_poten = None

    def allocate_internal_states(self, b_size, device):
        mem_shapes = []
        for i in range(len(hidden_sizes)):
            mem_shapes.append((b_size,hidden_sizes[i]))

        self.mem_poten_noiseless = []
        self.total_var = []
        self.spike_list = []
      

        for shape in mem_shapes:
            self.mem_poten_noiseless.append(torch.zeros(shape, device=device))
            self.total_var.append(torch.zeros(shape, device=device))
            self.spike_list.append(torch.zeros(shape, device=device))

        self.output_mem_poten = torch.zeros(b_size, self.output_size, device=device)
        self.output_mem_poten_sum = torch.zeros(b_size, self.output_size, device=device)


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

        for t in range(time_steps):
            data=data_in[:,t,:]
            for j in range(len(hidden_sizes)):

                if model_style == "bayesian":
                    if j % 2 == 1:
                        identity=self.spike_list[j - 1].clone()

                    e_sigscale = F.softplus(self.scales[j])
                    if torch.isnan(e_sigscale).any():
                        raise ValueError("NaN detected!")
                    
                    e_sigscale_rec=F.softplus(self.scales_rec[j])
                    e_sigscale_bias=F.softplus(self.scales_biases[j])

               
                    if j == 0:
                        added_var_in=(((data**2).sum(dim=1,keepdim=True))*(e_sigscale**2))

                    else:
                        added_var_in=(((self.spike_list[j-1].detach()**2).sum(dim=1,keepdim=True))*(e_sigscale**2))
                    

                    added_var_rec=(((self.spike_list[j].detach()**2).sum(dim=1,keepdim=True))*(e_sigscale_rec**2))
                    added_var = added_var_in+added_var_rec


                    added_var_mask = (added_var+e_sigscale_bias**2 > 0).float()
                    added_var_safe = torch.clamp(added_var+e_sigscale_bias**2, min=1e-6)
                    added_sd = added_var_mask * torch.sqrt(added_var_safe+e_sigscale_bias**2)

                    self.total_var[j] = (beta_mem[j]**2) * self.total_var[j] + added_var

                    total_var_mask = ((self.total_var[j]+e_sigscale_bias**2) > 0).float()
                    total_var_safe = torch.clamp((self.total_var[j]+e_sigscale_bias**2), min=1e-6)
                    self.total_sd = total_var_mask * torch.sqrt(total_var_safe)

                    added_noise = torch.tensor(0,device=device)
                    if noise_style == "noisy":
                        added_noise = (torch.normal(0, 1, size=self.total_sd.shape, device=device) * self.total_sd).detach()

                    if j == 0:
                        new_input=(torch.mm(data,self.weights[0])+torch.mm(self.spike_list[0],self.recurrent_weights[0]*self.masks[0])) *self.scalers[time_steps * j + t]+self.biases[time_steps * j + t]
                        self.mem_poten_noiseless[0] = beta_mem[0] * self.mem_poten_noiseless[0] +  new_input

                    else:
                        new_input=(torch.mm(self.spike_list[j-1],self.weights[j])+torch.mm(self.spike_list[j],self.recurrent_weights[j]*self.masks[j]))*self.scalers[time_steps * j + t]+self.biases[time_steps * j + t]
                        self.mem_poten_noiseless[j] = beta_mem[j] * self.mem_poten_noiseless[j] +new_input

                 
                    reg_mems.append(new_input)
                    reg_combined_sigscale.append(added_sd)

                    heaviside_input = self.mem_poten_noiseless[j] - self.thresholds[j]
                    self.spike_list[j] = Activation_Bayes.apply(heaviside_input, added_noise, self.total_sd)

                    self.mem_poten_noiseless[j]=self.mem_poten_noiseless[j]- self.thresholds[j] * self.spike_list[j]

                    if j % 2 == 0 and j != 0:
                        self.spike_list[j] = self.spike_list[j] + identity


                elif model_style == "sg":

                    if j % 2 == 1:
                        identity=self.spike_list[j - 1].clone()

                    if j == 0:
                        self.mem_poten_noiseless[0] = beta_mem[0] * self.mem_poten_noiseless[0] +  self.bn_layers[time_steps * j + t](torch.mm(data,self.weights[0])+torch.mm(self.spike_list[0],self.recurrent_weights[0]*self.masks[0]))

                    else:
                        self.mem_poten_noiseless[j] = beta_mem[j] * self.mem_poten_noiseless[j] +self.bn_layers[time_steps * j + t](torch.mm(self.spike_list[j-1],self.weights[j])+torch.mm(self.spike_list[j],self.recurrent_weights[j]*self.masks[j]))

                    total_noise = 0
                    heaviside_input = self.mem_poten_noiseless[j] - self.thresholds[j]
                    self.spike_list[j] = Activation_SG.apply(heaviside_input, total_noise)

                    self.mem_poten_noiseless[j]=self.mem_poten_noiseless[j]- self.thresholds[j] * self.spike_list[j]

                    if j % 2 == 0 and j != 0:
                        self.spike_list[j] = self.spike_list[j] + identity
       
            
            self.output_mem_poten = self.output_mem_poten + torch.mm(self.spike_list[-1],self.output_weights)


        output = self.output_mem_poten / time_steps
        return output, reg_mems, reg_combined_sigscale


# ---- Evaluation Function ----

def evaluate(model, test_loader,eval_device):
    model.eval()
    correct = 0
    total = 0
    test_loss = 0.0

    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(eval_device) 
            target = target.to(eval_device)

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


    return test_accuracy, test_avg_loss

# ---- Training Function ----
def adjust_learning_rate(optimiser, epoch, total_epochs):
   

    lr1_init = 0.005 
    lr2_init = 0.05 
    min_lr1 = lr1_init / 10    
    min_lr2 = lr2_init /10 


    lr1 = min_lr1 + (lr1_init - min_lr1) * (1 + math.cos(math.pi * (epoch) / total_epochs)) / 2
    lr2 = min_lr2 + (lr2_init - min_lr2) * (1 + math.cos(math.pi * (epoch) / total_epochs)) / 2

    
    optimiser.param_groups[0]['lr']=lr1
    optimiser.param_groups[1]['lr']=lr1
    optimiser.param_groups[2]['lr']=lr2 
    optimiser.param_groups[3]['lr']=lr2 
    optimiser.param_groups[4]['lr']=lr1
    optimiser.param_groups[5]['lr']=lr1
    optimiser.param_groups[6]['lr']=lr2
    optimiser.param_groups[7]['lr']=lr2
    optimiser.param_groups[8]['lr']=lr2



def train_model(model, train_loader, epochs, device):

    param_groups = [
    {'params': model.weights, 'lr': 0.001}, 
    {'params': model.output_weights, 'lr': 0.001}, 
    {'params': model.scales, 'lr': 0.01},
    {'params': model.scales_rec, 'lr': 0.01},
    {'params': model.recurrent_weights, 'lr': 0.001},
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
            if model_style == "bayesian":
                for i in range(len(reg_mems)):
                    mask = reg_combined_sigscale[i] > 0
                    safe_sigscale = torch.clamp(reg_combined_sigscale[i], min=1e-6)
                    ratio_squared = mask*(reg_mems[i] / safe_sigscale)**2
                    reg_loss += torch.mean(torch.sum(ratio_squared, dim=(1)))
       
            lambda_reg =1e-6
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

        test_accuracy, avg_test_loss = evaluate(model, test_loader, device)
        test_losses.append(avg_test_loss)
        accuracies.append(test_accuracy)


        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f} - Train Acc: {avg_train_accuracy:.4f}  - Test Loss: {avg_test_loss:.4f} - Test Acc: {test_accuracy:.4f}")

    return train_losses, test_losses, accuracies

# ---- Global Settings ----


output_size = 20
hidden_sizes=[256,256]


delta_t = 1


results = {}

# ---- Main ----

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default="bayesian")
    parser.add_argument("--scale_style", type=str, default="learnable")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs",     type=int, default=30)
    parser.add_argument("--time_steps",     type=int, default=100)
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

    train_loader, test_loader = get_dataloaders(batch_size=args.batch_size)

    method =  f"{args.method},{args.scale_style}"

    print(f"\nTraining method: {method}")
    start_time = time.time()

    model = CustomNet(
        input_size=700,
        hidden_sizes=hidden_sizes,
        output_size=output_size
    ).to(device)

    train_losses, test_losses, accuracies = train_model(model, train_loader, epochs=args.epochs, device=device)

    results[method] = {
        'train_loss': train_losses,
        'test_loss': test_losses,
        'accuracy': accuracies
    }

    end_time = time.time()
    print(f"\nTotal training time for {method}: {end_time - start_time:.2f} seconds")


    fname = f"{model_style}_shd.pth"

    checkpoint = {
    'model_state_dict': model.state_dict(),
    'results': results,
    }

    torch.save(checkpoint, fname)
        
