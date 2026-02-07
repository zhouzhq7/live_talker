"""
Live Talker Test Suite

测试目录结构:
- unit/: 单元测试
- integration/: 集成测试
- performance/: 性能测试
- acceptance/: 验收测试
- test_data/: 测试数据
- utils/: 测试工具

运行测试:
- 全部测试: pytest
- 特定 Phase: pytest -m phase1
- 特定类型: pytest -m unit
- 排除慢测试: pytest -m "not slow"
"""

__version__ = "1.0.0"
