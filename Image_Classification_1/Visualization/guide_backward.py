import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision
from torch.optim.lr_scheduler import StepLR
import imageio
# a helper method used for the visualiation
def setbyname(model, name, value):
    def iteratset(obj, components, value):
        print('components', components)
        if not hasattr(obj, components[0]):
            return False
        elif len(components) == 1:
            setattr(obj, components[0], value)
            return True
        else:
            nextobj = getattr(obj, components[0])
            return iteratset(nextobj, components[1:], value)

    components = name.split('.')
    success = iteratset(model, components, value)
    return success


class CustomReLU(torch.autograd.Function):
    # define two static methods here
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        grad_input[grad_output < 0] = 0
        return grad_input


class CustomReluModule(nn.Module):

    def __init__(self):
        super(CustomReluModule, self).__init__()

    def forward(self, input):
        return CustomReLU.apply(input)

    def extra_repr(self):
        repr_str = "Hello here is the Custom ReLU Module"
        return repr_str


class Guided_backprop():
    # initialize guided_backpropagation here
    def __init__(self, model, device):
        self.model = model
        self.image_reconstruction = None
        self.device = device
        self.activation_maps = []
        self.model.eval()
        # self.register_hooks()
        self.replace_modules()


    def replace_modules(self):
        def recursively_replace_modules(module, inherit_name=None):
        """
        take module as parameter, firstly put inherit_name and sub_name into target_name which is an array, 
        and then replace modules by using for loop
        """
            for sub_name, sub_module in module.named_children():
                target_name = []
                if inherit_name is not None:
                    target_name.append(inherit_name)
                target_name.append(sub_name)
                target_name = ".".join(target_name)
                if isinstance(sub_module, nn.Sequential):
                    recursively_replace_modules(sub_module, target_name)
                elif isinstance(sub_module, nn.ReLU):
                    success = setbyname(self.model, target_name, CustomReluModule())
                    print(f"replace {target_name} with CustomReLU {'success' if success else 'fail'}")

        recursively_replace_modules(self.model, None)

    def process(self, input_image, target_class):
        """
        the method used to visualize images
        """
        input_image.requires_grad_()
        model_output = self.model(input_image)
        self.model.zero_grad()
        pred_class = model_output.argmax().item()


        grad_target_map = torch.zeros(
            model_output.shape,
            dtype=torch.float
        ).to(self.device)

        if target_class is not None:
            grad_target_map[0][target_class] = 1
        else:
            grad_target_map[0][pred_class] = 1

        model_output.backward(grad_target_map)

        # result = self.image_reconstruction.data[0].permute(1, 2, 0)
        result = input_image.grad.data
        print(result.shape)
        return result


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    dataset1 = torchvision.datasets.CIFAR10(
        root='"E:/Machine Learning/project2/input/face-classifier/Images/"', train=False, download=True, transform=transform)
    kwargs = {'batch_size': 1}
    train_loader = torch.utils.data.DataLoader(dataset1, **kwargs)

    model = torch.load("cifar_model.pkl").to(device)
    guided_bp = Guided_backprop(model,device)

    show_how_many_images = 1
    for i in range(show_how_many_images):
        inputs = next(iter(train_loader))
        image = inputs[0]
        image = image.to(device)
        result = guided_bp.process(image, None)

        print("img", image.shape)
        image = inputs[0][0].permute(1, 2, 0)
        print("img_after", image.shape)
        imageio.imwrite('./Figure/test.jpg', image)

        result_draw = result[0].permute(1, 2, 0).cpu()
        imageio.imwrite('./Figure/test_grad.jpg', result_draw)

        print("result", result.shape)


    print('END')
