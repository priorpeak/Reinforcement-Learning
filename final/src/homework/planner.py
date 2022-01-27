# Collaborated with Allen Zou

import torch
import torch.nn.functional as F


def spatial_argmax(logit):
    """
    Compute the soft-argmax of a heatmap
    :param logit: A tensor of size BS x H x W
    :return: A tensor of size BS x 2 the soft-argmax in normalized coordinates (-1 .. 1)
    """
    weights = F.softmax(logit.view(logit.size(0), -1), dim=-1).view_as(logit)
    return torch.stack(((weights.sum(1) * torch.linspace(-1, 1, logit.size(2)).to(logit.device)[None]).sum(1),
                        (weights.sum(2) * torch.linspace(-1, 1, logit.size(1)).to(logit.device)[None]).sum(1)), 1)


class Planner(torch.nn.Module):
    def __init__(self):

      super().__init__()

      '''
      IMPORTANT, PLEASE READ:
      In order to run "python3 -m train" you must uncomment the cuda detection line and comment the line below it.
      In order to run "python3 -m planner..." you must revert back to the original state (Comment the cuda detection line and uncomment the line below it).
      '''

    #   self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #   self.device = "cpu"

      layers = []
    #   layers.append(torch.nn.Conv2d(3,16,5, 2, 2).to(self.device))
    #   layers.append(torch.nn.ReLU().to(self.device))
    #   layers.append(torch.nn.Conv2d(16,1,5, 2, 2).to(self.device))
    #   layers.append(torch.nn.ReLU().to(self.device))


      layers.append(torch.nn.Conv2d(3, 16, 5, 1, 2))
      layers.append(torch.nn.BatchNorm2d(16))
      layers.append(torch.nn.ReLU())
      layers.append(torch.nn.MaxPool2d(kernel_size=2, stride=2))
    #   layers.append(torch.nn.Dropout2d(p=0.25))

      layers.append(torch.nn.Conv2d(16, 32, 5, 1, 2))
      layers.append(torch.nn.BatchNorm2d(32))
      layers.append(torch.nn.ReLU())
      layers.append(torch.nn.MaxPool2d(kernel_size=2, stride=2))
    #   layers.append(torch.nn.Dropout2d(p=0.25))

      layers.append(torch.nn.Conv2d(32, 64, 5, 1, 2))
      layers.append(torch.nn.BatchNorm2d(64))
      layers.append(torch.nn.ReLU())
      layers.append(torch.nn.MaxPool2d(kernel_size=2, stride=2))
    #   layers.append(torch.nn.Dropout2d(p=0.25))

      layers.append(torch.nn.Conv2d(64, 128, 5, 1, 2))
      layers.append(torch.nn.BatchNorm2d(128))
      layers.append(torch.nn.ReLU())
      layers.append(torch.nn.MaxPool2d(kernel_size=2, stride=2))
    #   layers.append(torch.nn.Dropout2d(p=0.25))

      layers.append(torch.nn.Conv2d(128, 1, 5, 1, 2))
      layers.append(torch.nn.BatchNorm2d(1))
      layers.append(torch.nn.ReLU())
      layers.append(torch.nn.MaxPool2d(kernel_size=2, stride=2))
      # layers.append(torch.nn.Dropout2d(p=0.5))

      self._conv = torch.nn.Sequential(*layers)



    def forward(self, img):
        """
        Your code here
        Predict the aim point in image coordinate, given the supertuxkart image
        @img: (B,3,96,128)
        return (B,2)
        """

        x = self._conv(img)

        # print(img.shape)
        # print(x.shape)

        return spatial_argmax(x[:, 0])
        # return self.classifier(x.mean(dim=[-2, -1]))


def save_model(model):
    from torch import save
    from os import path
    if isinstance(model, Planner):
        return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'planner.th'))
    raise ValueError("model type '%s' not supported!" % str(type(model)))


def load_model():
    from torch import load
    from os import path
    r = Planner()
    # modified map_location to cuda:0
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), 'planner.th'), map_location='cuda:0'))
    return r


if __name__ == '__main__':
    from controller import control
    from utils import PyTux
    from argparse import ArgumentParser


    def test_planner(args):
        # Load model
        planner = load_model().eval()
        pytux = PyTux()
        for t in args.track:
            steps, how_far = pytux.rollout(t, control, planner=planner, max_frames=1000, verbose=args.verbose)
            print(steps, how_far)
        pytux.close()


    parser = ArgumentParser("Test the planner")
    parser.add_argument('track', nargs='+')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()
    test_planner(args)
