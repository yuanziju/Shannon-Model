# 测试指标清单

> 本文件定义代码测试的评估指标。**不能只看覆盖率**，必须多维评估。

---

## 1. 多维覆盖率

不看单一"覆盖率"数字，看三个维度：

| 维度 | 含义 | 工具 | 目标 |
|------|------|------|------|
| 行覆盖（Line） | 多少行代码被执行过 | `pytest --cov` / `coverage` | ≥80% |
| 分支覆盖（Branch） | 多少 if/else 分支被覆盖 | `coverage --branch` | ≥70% |
| 函数覆盖（Function/Method） | 多少函数被调用过 | `coverage` | ≥85% |

### 注意
- 100% 行覆盖不等于 100% 分支覆盖
- 分支覆盖比行覆盖更严格
- 函数覆盖能发现"有函数从没被调用过"

### 命令
```bash
# 多维覆盖率
pytest --cov=src --cov-branch --cov-report=term-missing

# 查看哪些函数没被覆盖
coverage report --skip-covered --show-missing
```

---

## 2. 边界覆盖

每个函数必须测试以下边界（如果适用）：

| 边界类型 | 测试输入 | 例子 |
|----------|----------|------|
| 空值 | `None` / `[]` / `""` / `{}` | `test_handle_empty_list` |
| 负数 | `-1` / `-0.01` | `test_calculate_with_negative` |
| 零 | `0` / `0.0` | `test_divide_by_zero` |
| 超长 | 10MB 字符串 / 100万元素列表 | `test_handle_large_input` |
| 极值 | `INT_MAX` / `INT_MIN` / `float('inf')` | `test_overflow` |
| 并发 | 多线程同时访问 | `test_thread_safety` |
| 非法输入 | 错误类型 / 损坏数据 | `test_invalid_input_raises` |
| 边界条件 | off-by-one / 边界值 | `test_off_by_one` |

### 检查清单
- [ ] 空值测了吗？
- [ ] 负数测了吗？
- [ ] 零测了吗？
- [ ] 超长输入测了吗？
- [ ] 非法输入测了抛异常吗？
- [ ] 边界值（off-by-one）测了吗？

---

## 3. 断言质量

避免伪测试（看着有测试，实际没测任何东西）。

### 伪测试反模式

#### 3.1 空 assert
```python
# 错：啥也没断言
def test_function():
    result = my_function()
    # 没有 assert

# 错：恒真断言
def test_function():
    result = my_function()
    assert True  # 废话
```

#### 3.2 只测 happy path
```python
# 错：只测正常情况
def test_login():
    assert login("user", "pass") is not None
    # 没测错误密码、不存在用户、空密码

# 对：测 happy + sad path
def test_login_success():
    assert login("user", "pass") == expected_token

def test_login_wrong_password():
    assert login("user", "wrong") is None

def test_login_nonexistent_user():
    assert login("nobody", "pass") is None

def test_login_empty_credentials():
    with pytest.raises(ValueError):
        login("", "")
```

#### 3.3 过宽断言
```python
# 错：断言太宽，啥都通过
def test_process():
    result = process()
    assert result is not None  # 太宽

# 对：断言具体值
def test_process():
    result = process()
    assert result.status == "ok"
    assert result.records == [1, 2, 3]
    assert result.elapsed < 1.0
```

#### 3.4 断言 mock 而非行为
```python
# 错：测 mock 被调用，没测实际行为
def test_save():
    with patch('db.save') as mock:
        save_user(user)
        mock.assert_called_once()  # 只测了调用，没测结果

# 对：测实际结果
def test_save_returns_id(test_db):
    user_id = save_user(user, db=test_db)
    assert user_id > 0
    assert test_db.get_user(user_id) == user
```

### 断言质量检查清单
- [ ] 每个测试都有有意义的 assert？
- [ ] assert 断言具体值，不是 `is not None` / `True`？
- [ ] 测了错误路径，不只测 happy path？
- [ ] 断言的是行为，不是 mock 调用？

---

## 4. 关键路径覆盖

不看测试总数，看关键路径有没有测试。

### 4.1 识别关键路径
- 核心业务逻辑（如支付、认证、订单）
- 高频调用路径（每次请求都走的代码）
- 错误处理路径（异常 / 超时 / 重试）
- 安全相关路径（权限检查 / 输入验证）

### 4.2 关键路径必须有测试
```python
# 支付核心逻辑——必须有测试
def test_payment_charges_correct_amount():
    ...

def test_payment_fails_when_insufficient_balance():
    ...

def test_payment_idempotent_on_retry():
    ...

# 认证核心逻辑——必须有测试
def test_auth_rejects_invalid_token():
    ...

def test_auth_expires_after_timeout():
    ...
```

### 4.3 非关键路径可以少测
- 日志记录
- 监控上报
- 调试代码
- 实验性功能

---

## 5. 综合评估

完工时回答这 5 个问题：

1. **多维覆盖率**：行/分支/函数覆盖率分别多少？达标吗？
2. **边界覆盖**：每个函数的边界都测了吗？（空/负/零/超长/非法）
3. **断言质量**：有没有伪测试？有没有只测 happy path？
4. **关键路径**：核心逻辑都有测试吗？
5. **测试独立性**：测试之间互不依赖吗？随机顺序跑能过吗？

**任意一项不达标，不能进入子代理审查步骤。**
