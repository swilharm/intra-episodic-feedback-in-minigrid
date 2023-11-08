from typing import List

import torch
import torch.nn as nn
import gymnasium
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from matplotlib import pyplot as plt

from feature_extraction.film import FiLM


class CombinedFeaturesExtractor(BaseFeaturesExtractor):
    """Applies observation-specific feature extractors,
    then uses a FiLM layer to condition the image embedding on the text"""
    def __init__(self, observation_space: gymnasium.Space, **kwargs):
        super().__init__(observation_space, kwargs['film_dim'])

        self.vision = VisionFeaturesExtractor(observation_space.spaces['image'], **kwargs)
        self.language = LanguageFeaturesExtractor(observation_space.spaces['mission'], **kwargs)
        self.direction = DirectionFeaturesExtractor(observation_space.spaces['direction'], **kwargs)

        self.film_controller = FiLM(kwargs['language_dim'], kwargs['film_dim'],
                                    kwargs['vision_dim'], kwargs['film_dim'])
        self.film_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.film_norm = nn.LayerNorm(kwargs['film_dim'])

    def forward(self, observations: torch.Tensor):
        # self.__show_frames(observations)
        vision_embeddings = self.vision(observations['image'])
        language_embeddings = self.language(observations['mission'])
        direction_embeddings = self.direction(observations['direction'])

        filmed_vision_embeddings = self.film_controller(vision_embeddings, language_embeddings)
        filmed_vision_embeddings = filmed_vision_embeddings + vision_embeddings  # residual connection
        filmed_vision_embeddings = self.film_pool(filmed_vision_embeddings)
        filmed_vision_embeddings = filmed_vision_embeddings.reshape(filmed_vision_embeddings.shape[0], -1)
        vision_embeddings = self.film_norm(filmed_vision_embeddings)

        return (vision_embeddings + direction_embeddings) / 2

    def __show_frames(self, observations: torch.Tensor):
        """Displays the stacked observations with pyplot"""
        from util.wrappers import W2I, I2W
        I2W[0] = ''
        fig, axs = plt.subplots(1, 3)
        fig: Figure
        axs: List[Axes]
        plt.sca(ax=axs[0])
        plt.gca().set_title(' '.join((I2W[int(idx)] for idx in observations['mission'][0][0:6])).strip())
        plt.imshow(torch.permute(observations['image'][0][0:3], (1, 2, 0)))
        plt.sca(ax=axs[1])
        plt.gca().set_title(' '.join((I2W[int(idx)] for idx in observations['mission'][0][6:12])).strip())
        plt.imshow(torch.permute(observations['image'][0][3:6], (1, 2, 0)))
        plt.sca(ax=axs[2])
        plt.gca().set_title(' '.join((I2W[int(idx)] for idx in observations['mission'][0][12:18])).strip())
        plt.imshow(torch.permute(observations['image'][0][6:9], (1, 2, 0)))
        fig.tight_layout()
        plt.show()


class VisionFeaturesExtractor(BaseFeaturesExtractor):
    """Vision feature extractor matching Sadler et al., 2023"""
    def __init__(self,
                 observation_space: gymnasium.spaces.Box,
                 vision_dim: int,
                 **kwargs,
                 ) -> None:
        super().__init__(observation_space, vision_dim)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=n_input_channels, out_channels=vision_dim, kernel_size=1, padding=0),
            nn.BatchNorm2d(vision_dim),
            nn.ReLU(),
            nn.Conv2d(in_channels=vision_dim, out_channels=vision_dim, kernel_size=3, padding="same"),
            nn.BatchNorm2d(vision_dim),
            nn.ReLU(),
            nn.Conv2d(in_channels=vision_dim, out_channels=vision_dim, kernel_size=3, padding="same"),
            nn.BatchNorm2d(vision_dim),
            nn.ReLU(),
            nn.Conv2d(in_channels=vision_dim, out_channels=vision_dim, kernel_size=3, padding="same"),
            nn.BatchNorm2d(vision_dim),
            nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.cnn(observations)


class LanguageFeaturesExtractor(BaseFeaturesExtractor):
    """Language feature extractor matching Sadler et al., 2023"""
    def __init__(self,
                 observation_space: gymnasium.spaces.Box,
                 embedding_dim: int,
                 language_dim: int,
                 **kwargs,
                 ) -> None:
        super().__init__(observation_space, language_dim)
        vocab_size = observation_space.high[0]
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.rnn = nn.GRU(embedding_dim, language_dim, batch_first=True)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(observations.int())
        _, hidden = self.rnn(embedded)
        return hidden[-1]


class DirectionFeaturesExtractor(BaseFeaturesExtractor):
    """Feature extractor for the direction, applies linear transformation to input"""
    def __init__(self,
                 observation_space: gymnasium.spaces.Box,
                 direction_dim: int,
                 **kwargs,
                 ) -> None:
        super().__init__(observation_space, direction_dim)
        in_features = observation_space.shape[0]
        self.linear = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=direction_dim),
            nn.LayerNorm(direction_dim)
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(observations)
