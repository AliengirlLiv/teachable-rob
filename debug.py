import torch.nn as nn
import math
import torch.nn.functional as F
import torch
import torchvision.transforms as transforms
import torchvision
from rlbench.environment import Environment
from rlbench.action_modes import ArmActionMode, ActionMode
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import ReachTarget
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import gym
import rlbench.gym
import wandb
import argparse


# SOURCE: https://pytorch.org/vision/stable/models.html

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, input_groups=3):
        self.inplanes = 60
        super(ResNet, self).__init__()
        hidden_channels = 60  # chosen b/c it's divisible by 1, 2, 3, 4, 5, 6 so we can use multiple # inputs
        self.conv1 = nn.Conv2d(input_groups*3, hidden_channels, kernel_size=7, stride=2, padding=3,
                               bias=False, groups=input_groups)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, hidden_channels, layers[0], groups=input_groups)
        self.layer2 = self._make_layer(block, hidden_channels*2, layers[1], stride=2, groups=input_groups)
        self.layer3 = self._make_layer(block, hidden_channels*4, layers[2], stride=2, groups=input_groups)
        self.layer4 = self._make_layer(block, hidden_channels*8, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(hidden_channels * 8 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, groups=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False, groups=groups),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


class ImitationLearning(nn.Module):  # TODO: actually have a policy in here!

    def __init__(self, num_cameras, action_dim):
        super().__init__()
        self.model = resnet18(num_classes=action_dim, input_groups=num_cameras)

    def forward(self, batch):
        return self.model(batch)

    def behaviour_cloning_loss(self, ground_truth_actions, predicted_actions):
        return F.mse_loss(ground_truth_actions[:, :3], predicted_actions[:, :3])

def load_env(path, task_str):
    obs_config = ObservationConfig()
    obs_config.set_all(True)
    action_mode = ActionMode(ArmActionMode.DELTA_EE_POSE_PLAN_WORLD_FRAME)
    env = Environment(action_mode, str(path), obs_config, False)
    env.launch()
    return env.get_task(task_str), env

def load_data(path, variations, task_str, val_split=0, num_demos=10):
    full_path = pathlib.Path(path)
    demos_train = []
    demos_val = []
    env, big_env = load_env(full_path, task_str)
    for v in variations:         
        env.set_variation(v)
        task_demos = env.get_demos(num_demos, live_demos=False)  # -> List[List[Observation]]
        split_index = round(val_split * len(task_demos))
        demos_train += task_demos[split_index:]
        demos_val += task_demos[:split_index]
    big_env.shutdown()
    return np.concatenate([np.array(d) for d in demos_train]), np.concatenate([np.array(d) for d in demos_val])

def evaluate(env_path, task_str, policy, num_rollouts=10, horizon=40):
    print("about to load env")
    env, big_env = load_env(env_path, task_str)
    rewards = []
    successes = []
    for i in range(num_rollouts):
        _, obs = env.reset()
        for t in range(horizon):
            batch = preprocess([obs])
            action = policy(batch).detach().cpu().numpy()[0]
            # Add in an extra element for the arm
            a_min = np.array([ 2.06e-1, -8.1e-3,  1.224, -1.5e-4, 1.0e-1, -1.4e-4,  1.7e-5])
            a_max = np.array([ 2.78e-1,  9.17e-2,  1.47e+00,  8.296e-06, 1.0, -2.214e-07,  1.208e-01])
            action = np.clip(action, a_min, a_max)
            # Keep the quaternion constant (TODO: later keep the default?)
            action[3] = 1
            quat = np.array([9.96596654e-01, -4.51784236e-05,  7.15277889e-02])
            quat[-1] = np.sqrt(1 - np.sum(quat[:2]**2))  # quaternions must be unit
            action[4:7] = quat
            action = np.concatenate([action, np.array([0])])
            print("our action", action)
            obs, reward, done = env.step(action)
            print("=============================================")
            rewards.append(reward)
            if t == horizon - 1:
                successes.append(done)
    big_env.shutdown()
    print("close env")
    return rewards, successes

def preprocess_gym(obs):
    o = torch.FloatTensor([np.concatenate([obs['left_shoulder_rgb'], obs['right_shoulder_rgb'],
        obs['front_rgb'], obs['wrist_rgb']], axis=-1) ]).cuda()
    print("SH", o.shape)
    return o
    
def preprocess(batch):
    batch_images = torch.FloatTensor([np.concatenate([
                                        obs.left_shoulder_rgb,
                                        obs.right_shoulder_rgb,
                                        obs.front_rgb,
                                        obs.wrist_rgb], axis=-1)
                                    for obs in batch]).cuda()
    batch_images = torch.movedim(batch_images, -1, 1)  # BCHW
    batch_images = augmenter(batch_images)
    return batch_images

def compute_loss(dataset, policy):
    batch = np.random.choice(dataset, size=batch_size, replace=True)
    batch_images = preprocess(batch)
    predicted_actions = il(batch_images)
    ground_truth_actions = torch.FloatTensor([obs.gripper_pose for obs in batch]).cuda()
    loss = il.behaviour_cloning_loss(ground_truth_actions, predicted_actions)
    return loss

# TODO: we should be able to get num_demos and action_dim from the data
num_demos = 2
batch_size = 16
num_cameras = 4
action_dim = 7
val_split = .5
dataset_path = 'reach_target_2_pose'
task = ReachTarget
log_wandb = True
n_itr = 1000000
eval_every = 100
run_name = 'WHATEVER'
task_name = 'reach_target'
horizon = 40
base_path = '/home/olivia/Teachable/rlbench_data/'


il = ImitationLearning(num_cameras, action_dim).cuda()

train_dataset, val_dataset = load_data(base_path + dataset_path, [0], task, val_split, num_demos)
test_dataset, _ = load_data(base_path + dataset_path, [18], task, val_split, num_demos)

wandb.init(project='rlbench', mode='online' if log_wandb else 'disabled')
wandb.run.name = run_name
wandb.config.num_demos = len(train_dataset)
wandb.config.dataset = dataset_path
wandb.config.task = task_name

optimizer = torch.optim.Adam(il.parameters(), lr=1e-3)
augmenter = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(.75, 1), ratio=(1,1)),
        torchvision.transforms.ColorJitter(),
    ])

# An example of using the demos to 'train' using behaviour cloning loss.
val_loss = torch.zeros(1) - 1
for i in range(n_itr):
    print("starting itr", i)
    if (i % eval_every == 0) or (i == n_itr - 1):
        il.eval()
        with torch.no_grad():
            print("validation")
            val_loss = compute_loss(val_dataset, il)
            wandb.log({'val/loss': val_loss}, step=i)

            print("testing")
            test_loss = compute_loss(test_dataset, il)
            wandb.log({'test/loss': test_loss}, step=i)

            print("evaluation")
            rewards, successes = evaluate(base_path + dataset_path, task, il, horizon=horizon)
            wandb.log({'eval/reward': np.mean(rewards)}, step=i)
            wandb.log({'eval/success': np.mean(successes)}, step=i)

    il.train()
    print("training")
    loss = compute_loss(train_dataset, il)
    wandb.log({'val/loss': loss}, step=i)
    print("'training' iteration %d" % i, loss.item(), val_loss.item())
    wandb.log({'train/loss': loss}, step=i)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


