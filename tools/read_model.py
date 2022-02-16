import torch
from os import path
import os
import sys
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
print(path.dirname(path.dirname(path.abspath(__file__))))

from opts import get_args
from model import UNet, SegNet, DenseNet

args = get_args()

n_channels = args.channels
n_classes = args.classes

model = None

if args.model == 'UNet':
    model = UNet(in_channels=n_channels, n_classes=n_classes)

elif args.model == 'SegNet':
    model = SegNet(in_channels=n_channels, n_classes=n_classes)

elif args.model == 'DenseNet':
    model = DenseNet(in_channels=n_channels, n_classes=n_classes)
else:
    print("wrong model : must be UNet, SegNet, or DenseNet")
    # raise SystemExit

# summary(model, input_size=(n_channels, args.height, args.width), device = 'cpu')
print('model', args.model)
model_path = os.path.join(args.model_path, args.data_path, args.model + '.pth')
out_model_path = os.path.join(args.model_path, args.data_path, args.model + '.pt')
model.load_state_dict(torch.load(model_path))
model_scripted = torch.jit.script(model) # Export to TorchScript
model_scripted.save(out_model_path) # Save