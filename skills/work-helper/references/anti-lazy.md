# 防偷懒铁律

> 本文件定义"偷懒"的具体表现和检测方法。**绝对不能偷懒**——这是本 Skill 的第一铁律。

---

## 什么是偷懒

偷懒 = AI 在实现代码时，给出的不是完整实现，而是某种"占位"或"简化"版本，导致用户拿到的代码不能真正干活。

---

## 偷懒的 7 种表现

### 1. 空实现
```python
# 错：pass 不是实现
def login(username, password):
    pass

# 错：return None 不是实现
def calculate_total(items):
    return None

# 错：... 不是实现（Python 3 的 Ellipsis）
def process(data):
    ...
```

### 2. 占位实现
```python
# 错：硬编码假数据
def get_users():
    return [{"name": "Alice"}, {"name": "Bob"}]  # 假数据，不是真实实现

# 错：永远返回固定值
def calculate_tax(income):
    return 0  # 偷懒，没真正算

# 错：返回空集合假装"没有"
def search(query):
    return []  # 没真搜，直接返回空
```

### 3. 简化实现（用户要 A，只给 30%）
```python
# 用户要：实现一个支持分页、排序、过滤的用户查询接口
# 错：只实现最基础的，砍掉分页/排序/过滤
def get_users():
    return User.objects.all()  # 没分页、没排序、没过滤

# 对：完整实现
def get_users(page=1, per_page=20, sort_by="created_at", order="desc", filter=None):
    query = User.objects.all()
    if filter:
        query = query.filter(**filter)
    query = query.order_by(f"{'-' if order == 'desc' else ''}{sort_by}")
    return query.offset((page - 1) * per_page).limit(per_page)
```

### 4. 伪 TODO（用 TODO 逃避实现）
```python
# 错：明明能实现，却打 TODO
def send_email(to, subject, body):
    # TODO: implement email sending
    pass

# 对：要么真实现，要么 TODO 要有具体原因和计划
def send_email(to, subject, body):
    # TODO: 需要 SMTP 凭据，等运维提供后补全
    # 临时方案：写入 outbox 表，由后台 worker 异步发送
    Outbox.create(to=to, subject=subject, body=body)
    logger.info(f"Email queued for {to}")
```

### 5. 注释代替实现
```python
# 错：用注释描述应该做什么，但不写代码
def validate_email(email):
    # 这里应该用正则校验邮箱格式
    # 如果不合法抛 ValueError
    return True  # 永远返回 True，啥也没校验

# 对：真校验
import re

def validate_email(email):
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if not re.match(pattern, email):
        raise ValueError(f"Invalid email: {email}")
    return True
```

### 6. 抄测试框架的测试
```python
# 错：测试只断言能 import，没测任何行为
def test_module():
    import my_module  # noqa
    assert True

# 对：测实际行为
def test_login_returns_token_on_valid_credentials():
    token = login("user", "pass")
    assert token is not None
    assert len(token) > 0
```

### 7. 过度抽象（用抽象逃避实现）
```python
# 错：搞一堆抽象基类、接口、工厂，但具体实现是空的
class IUserRepository(ABC):
    @abstractmethod
    def get(self, id): pass

class UserRepository(IUserRepository):
    def get(self, id):
        pass  # 还是空的！

# 对：先有具体实现，再有抽象（YAGNI）
class UserRepository:
    def get(self, id):
        return self.db.query(User).filter_by(id=id).first()
```

---

## 合法的 TODO/FIXME

只有以下情况可以打 TODO/FIXME：

### 1. 依赖外部条件未就绪
```python
# TODO: 需要 SSO 服务地址，等运维提供后补全
# 当前先用本地认证兜底
def sso_login(token):
    return local_login(token)
```

### 2. 性能优化留待后续
```python
# FIXME: 这里 O(n²) 循环，数据量大时慢
# 后续用 hash table 优化，见 task T-XX
def find_duplicates(items):
    return [x for x in items if items.count(x) > 1]
```

### 3. 已知 bug 临时绕过
```python
# FIXME: 第三方库 xxx 有 bug，等他们修复
# 临时绕过：手动处理边界
# See: https://github.com/xxx/issues/123
def workaround(items):
    if len(items) == 0:
        return []  # 库 bug：空列表会崩
    return library_func(items)
```

### TODO/FIXME 规范
每个 TODO/FIXME 必须包含：
1. **原因**：为什么没实现
2. **计划**：后续怎么做
3. **临时方案**（如有）：当前怎么兜底

格式：
```python
# TODO: <原因>
# 计划: <后续怎么做>
# 临时: <当前兜底方案>
```

---

## 检测偷懒的自检清单

实现完成后，问自己：

- [ ] 有没有 `pass` / `return None` / `...` 当实现？
- [ ] 有没有硬编码假数据冒充真实数据？
- [ ] 用户要的功能，我都实现了吗？还是只实现了简单部分？
- [ ] 我的 TODO 都有具体原因和计划吗？
- [ ] 注释里的"应该做"，我都做了吗？
- [ ] 测试真的测了行为，还是只测了 import？
- [ ] 抽象层下面有没有真实实现？

**任意一项不达标，必须回去补全，不能进入子代理审查步骤。**

---

## 子代理审查重点

子代理审查时，**防偷懒是第一审查项**：
1. 扫描所有新代码，找 `pass` / `return None` / `...`
2. 扫描所有 TODO/FIXME，检查是否有原因和计划
3. 对照 spec 检查实现是否完整
4. 检查测试是否真测了行为

详见 [review-checklist.md](review-checklist.md)
