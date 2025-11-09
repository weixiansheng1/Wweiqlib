# -*- coding: utf-8 -*-
"""
Created on Sun Nov  9 15:06:22 2025

@author: 11298
"""

import qlib
from qlib.data import D

# 初始化 QLib（指定数据路径，若未自定义则无需传参）
qlib.init(provider_uri="~/.qlib/qlib_data/cn_data")  # 替换为你的数据路径

# 查看市场日历（验证数据是否加载）
print(D.calendar(start_time="2020-01-01", end_time="2020-01-10"))