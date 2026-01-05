"""
激活函数模块

这个模块包含了 YOLO 模型中使用的各种激活函数。
激活函数在神经网络中引入非线性，使模型能够学习复杂的模式。

主要激活函数:
    - AGLU: 自适应门控线性单元，具有可学习参数的统一激活函数
"""

# 导入 PyTorch 核心库
import torch  # PyTorch 深度学习框架
import torch.nn as nn  # PyTorch 神经网络模块


class AGLU(nn.Module):
    """来自 AGLU 的统一激活函数模块

    该类基于 AGLU (自适应门控线性单元) 方法实现了一个参数化的激活函数,具有可学习的 lambda 和 kappa 参数。

    属性:
        act (nn.Softplus): 具有负 beta 的 Softplus 激活函数
        lambd (nn.Parameter): 使用均匀分布初始化的可学习 lambda 参数
        kappa (nn.Parameter): 使用均匀分布初始化的可学习 kappa 参数

    方法:
        forward: 计算统一激活函数的前向传播

    示例:
        >>> import torch
        >>> m = AGLU()
        >>> input = torch.randn(2)
        >>> output = m(input)
        >>> print(output.shape)
        torch.Size([2])

    参考:
        https://github.com/kostas1515/AGLU
    """

    def __init__(self, device=None, dtype=None) -> None:
        """
        初始化统一激活函数，包含可学习参数

        AGLU 通过学习 lambda 和 kappa 参数来自适应调整激活函数的形状，
        使其能够在训练过程中适应不同的数据分布和任务需求。

        Args:
            device: 参数所在设备（CPU/GPU）
            dtype: 参数的数据类型
        """
        super().__init__()
        # Softplus 激活函数，beta=-1.0 表示使用负斜率
        self.act = nn.Softplus(beta=-1.0)
        # lambda 参数：控制激活函数的缩放，使用均匀分布初始化
        self.lambd = nn.Parameter(nn.init.uniform_(torch.empty(1, device=device, dtype=dtype)))
        # kappa 参数：控制激活函数的形状，使用均匀分布初始化
        self.kappa = nn.Parameter(nn.init.uniform_(torch.empty(1, device=device, dtype=dtype)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        应用自适应门控线性单元 (AGLU) 激活函数

        前向传播实现了 AGLU 激活函数，该函数使用可学习的 lambda 和 kappa 参数。
        函数应用一个变换，自适应地组合线性和非线性分量。

        数学公式:
            AGLU(x) = exp((1/λ) * Softplus(κ*x - log(λ)))

        其中:
            - λ (lambda): 缩放参数，控制函数的整体缩放
            - κ (kappa): 形状参数，控制函数的非线性程度
            - Softplus: 平滑的 ReLU 近似函数

        Args:
            x (torch.Tensor): 输入张量，将应用激活函数

        Returns:
            (torch.Tensor): 应用 AGLU 激活函数后的输出张量，形状与输入相同
        """
        # 将 lambda 限制在最小值 0.0001，避免除零错误
        lam = torch.clamp(self.lambd, min=0.0001)
        # 应用 AGLU 公式: exp((1/λ) * Softplus(κ*x - log(λ)))
        return torch.exp((1 / lam) * self.act((self.kappa * x) - torch.log(lam)))
