"""
Các chiến lược incremental learning trong project.

Module này chỉ làm nhiệm vụ gom và export các trainer/aggregator
cho những thuật toán FCIL chính:
- CGoFed: ràng buộc gradient theo không gian biểu diễn cũ
- FedCBDR: replay cân bằng theo lớp
- DER: mở rộng biểu diễn động theo từng task
- NICE: replay-free, cảm hứng từ neurogenesis
- GLFC: bù quên cục bộ + toàn cục
- Re-Fed: replay hiệu quả dựa trên PIM
"""

from .cgofed import CGoFedTrainer, CGoFedAggregator
from .fedcbdr import FedCBDRTrainer, FedCBDRAggregator
from .der import DERTrainer, DERAggregator
from .nice import NICETrainer, NICEAggregator
from .glfc import GLFCTrainer, GLFCAggregator
from .refed import ReFedTrainer, ReFedAggregator

__all__ = [
    "CGoFedTrainer",
    "CGoFedAggregator",
    "FedCBDRTrainer",
    "FedCBDRAggregator",
    "DERTrainer",
    "DERAggregator",
    "NICETrainer",
    "NICEAggregator",
    "GLFCTrainer",
    "GLFCAggregator",
    "ReFedTrainer",
    "ReFedAggregator",
]
