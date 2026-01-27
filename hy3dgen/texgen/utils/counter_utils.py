# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the respective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.


class RunningStats():
    def __init__(self) -> None:
        self.count = 0
        self.sum = 0
        self.mean = 0
        self.min = None
        self.max = None

    def add_value(self, value):
        self.count += 1
        self.sum += value
        self.mean = self.sum / self.count

        if self.min is None or value < self.min:
            self.min = value

        if self.max is None or value > self.max:
            self.max = value

    def get_count(self):
        return self.count

    def get_sum(self):
        return self.sum

    def get_mean(self):
        return self.mean

    def get_min(self):
        return self.min

    def get_max(self):
        return self.max
