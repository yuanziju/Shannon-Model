# TDD 规则 + 例外清单

> 本文件定义 TDD 的执行规则和例外情况。核心原则：完整 TDD，但不过度 TDD。

---

## 完整 TDD 流程（RED-GREEN-REFACTOR）

### 轻任务的"简化 TDD"指什么

重任务走完整 RED-GREEN-REFACTOR。轻任务的"简化"指：
- **bug fix**：至少先写**复现 bug 的测试**（RED），再修（GREEN），可省略独立 REFACTOR 步骤
- **小重构**：先确保现有测试覆盖了行为，重构后跑测试确认行为不变
- **补测试**：本身就是写测试，不存在 TDD 流程

注意："简化"≠"省略"。轻任务 bug fix **必须**先写复现测试，证明 bug 存在，否则不能进入修复步骤。

### RED：先写失败的测试

#### 新功能
- 写覆盖目标行为的测试
- 测试必须失败（证明功能还没实现）
- 测试名要描述行为：`test_login_returns_jwt_on_valid_credentials`

#### Bug fix
- **先写复现 bug 的测试**
- 测试必须失败（证明 bug 存在）
- 测试名要描述 bug：`test_login_returns_500_when_password_is_none`

### GREEN：写最小实现让测试通过
- 写最少的代码让测试通过
- 不要提前优化
- 不要加测试没要求的功能

### REFACTOR：重构
- 测试必须仍然通过
- 不改行为，只改结构
- 每改一点就跑一次测试

---

## 例外清单（不走 TDD）

以下情况**不走 TDD**，直接写代码：

### 1. 文档改动
- 只改 `.md` / `.rst` / `.txt` 文件
- 只改代码注释（不改代码逻辑）
- 改 README / CHANGELOG / API 文档

### 2. 配置改动
- 只改 `.yml` / `.yaml` / `.toml` / `.json` / `.ini` 配置
- 改 `pyproject.toml` 的依赖声明
- 改 `.github/workflows/*.yml` CI 配置
- 改 Dockerfile / docker-compose

### 3. 原型探索
- 用户明确说"先试一下""探索性""spike"
- 用户明确说"先 MVP""先简版"
- 注意：原型探索结束后，正式实现时必须走 TDD

### 4. 纯样式改动
- 格式化（black / prettier / rustfmt）
- import 排序（isort / reorder-python-imports）
- 改缩进 / 改引号风格
- 不改任何逻辑

---

## 不过度 TDD（重要）

### 不给测试写测试
- 不测 `pytest` / `unittest` 框架本身
- 不测 `mock` / `fixture` / `parametrize`
- 测试的质量靠 review，不靠递归测试

### 不给测试工具写测试
- 不测自己写的 test helper
- 不测 conftest.py 里的 fixture
- 不测 mock factory

### 不追求 100% 覆盖率
- 100% 覆盖率不等于好测试
- 例外清单内的代码不需要测试
- 生成的代码（如 protobuf / dataclass）不需要测
- 关键路径覆盖 > 100% 覆盖率

### 边界
- 一个函数 1-3 个测试（happy path + 边界 + 错误）
- 不要为每个 if 分支都写一个测试
- 用 `parametrize` 合并相似测试，不要复制粘贴

---

## TDD 反模式（必须避免）

### 1. 测试实现细节
```python
# 错：测实现细节（私有方法）
def test_helper_function_internal_state():
    assert obj._helper() == "xxx"

# 对：测行为
def test_public_api_returns_expected_result():
    assert obj.public_method() == "expected"
```

### 2. 测试耦合实现
```python
# 错：测试依赖实现细节（mock 内部调用）
def test_process():
    with patch('module.internal_func') as mock:
        process()
        mock.assert_called_once()

# 对：测试输入输出
def test_process_returns_expected():
    result = process(input="test")
    assert result.status == "ok"
```

### 3. 过度 mock
```python
# 错：mock 所有人，测了个寂寞
def test_service():
    with patch('db'), patch('cache'), patch('logger'):
        result = service()
        assert result is not None  # 啥也没测

# 对：用真实依赖或测试替身，测实际行为
def test_service_with_test_db(test_db):
    result = service(db=test_db)
    assert result.records == [...]
```

### 4. 测试噪音
```python
# 错：每个 assert 一个测试
def test_a():
    assert f(1) == 1
def test_b():
    assert f(2) == 2
def test_c():
    assert f(3) == 3

# 对：parametrize
@pytest.mark.parametrize("input,expected", [(1,1), (2,2), (3,3)])
def test_f(input, expected):
    assert f(input) == expected
```

---

## TDD 检查清单

写测试前问自己：
- [ ] 测试名描述行为还是实现？
- [ ] 测试是独立的吗（不依赖其他测试）？
- [ ] 测试是确定性的吗（不依赖时间/网络/随机）？
- [ ] 测试失败时能快速定位问题吗？
- [ ] 我在测行为还是在测 mock？
- [ ] 这个测试是不是给测试写测试（过度 TDD）？
