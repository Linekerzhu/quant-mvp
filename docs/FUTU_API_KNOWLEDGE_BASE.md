# Futu OpenAPI 知识库

> 本文档汇总 Futu OpenAPI 的关键信息，供项目开发参考。
> 官方文档：https://openapi.futunn.com/futu-api-doc/

---

## 1. 架构概述

Futu OpenAPI 由两部分组成：

- **OpenD**: 网关程序，运行在本地或云端，负责中转协议请求到富途后台
- **Futu API**: Python/Java/C#/C++/JavaScript SDK，封装了底层协议

```
[Your Strategy] --TCP--> [OpenD] --HTTPS--> [Futu Servers] --[Exchanges]
                :11111
```

### 工作流程

1. 安装并启动 OpenD 网关
2. 使用 Futu API SDK 连接 OpenD（默认端口 11111）
3. 通过 OpenD 获取行情、下单交易

---

## 2. 账号体系

### 平台账号
- 牛牛号（用户ID）
- 用于登录 OpenD 和获取行情

### 综合账户
- 支持多币种、多市场交易
- 综合账户-证券：股票、ETF、期权
- 综合账户-期货：期货产品

---

## 3. 支持的市场与品种

### 美股（US Market）

| 品种 | 行情 | 模拟交易 | 实盘交易 |
|------|------|----------|----------|
| 股票、ETFs | ✅ | ✅ | ✅ |
| 期权 | ✅ | ✅ | ✅ |
| 期货 | ✅ | ✅ | ✅ |

### 其他市场
- 港股（HK）：股票、窝轮、牛熊、期权、期货
- A股（A-Share）：A股通股票
- 新加坡、日本、澳大利亚（部分支持）

---

## 4. 核心 API 接口

### 交易接口 (OpenSecTradeContext)

```python
from futu import OpenSecTradeContext, TrdMarket, TrdEnv, TrdSide

# 初始化交易上下文
trade_ctx = OpenSecTradeContext(
    filter_trdmarket=TrdMarket.US,
    host='127.0.0.1',
    port=11111
)

# 解锁交易（实盘必需）
trade_ctx.unlock_trade(password_md5='YOUR_MD5_PASSWORD')

# 下单
ret, data = trade_ctx.place_order(
    price=150.0,                    # 限价
    qty=100,                        # 股数
    code='US.AAPL',                 # 股票代码
    trd_side=TrdSide.BUY,           # BUY or SELL
    order_type=OrderType.NORMAL,    # 限价单
    trd_env=TrdEnv.SIMULATE,        # SIMULATE or REAL
    time_in_force=TimeInForce.DAY,  # 当日有效
    fill_outside_rth=True,          # 允许盘前盘后
    remark='intent:12345'           # 幂等键（64字节限制）
)

# 查询订单
ret, data = trade_ctx.order_list_query(order_id='12345')

# 撤单
from futu import ModifyOrderOp
trade_ctx.modify_order(
    modify_order_op=ModifyOrderOp.CANCEL,
    order_id='12345',
    qty=0,
    price=0
)

# 查询持仓
ret, data = trade_ctx.position_list_query()

# 查询账户资金
ret, data = trade_ctx.accinfo_query()
```

### 行情接口 (OpenQuoteContext)

```python
from futu import OpenQuoteContext, SubType

# 初始化行情上下文
quote_ctx = OpenQuoteContext(host='127.0.0.1', port=11111)

# 订阅实时行情
quote_ctx.subscribe(['US.AAPL', 'US.TSLA'], [SubType.QUOTE, SubType.ORDER_BOOK])

# 获取实时报价
ret, data = quote_ctx.get_stock_quote(['US.AAPL'])

# 获取摆盘数据（bid/ask）
ret, data = quote_ctx.get_order_book('US.AAPL')
# data['Bid'] = [[bid_price, bid_volume, bid_order_count], ...]
# data['Ask'] = [[ask_price, ask_volume, ask_order_count], ...]

# 获取历史K线
from futu import KLType
ret, data = quote_ctx.get_history_klines(
    'US.AAPL',
    start='2024-01-01',
    end='2024-12-31',
    ktype=KLType.K_DAY
)

# 取消订阅
quote_ctx.unsubscribe(['US.AAPL'], [SubType.ORDER_BOOK])
```

---

## 5. 股票代码格式

### 代码格式规范

| 市场 | Futu 格式 | 示例 |
|------|-----------|------|
| 美股 | `US.{TICKER}` | `US.AAPL`, `US.TSLA` |
| 港股 | `HK.{CODE}` | `HK.00700`, `HK.09988` |
| A股 | `SH.{CODE}` / `SZ.{CODE}` | `SH.600519`, `SZ.000858` |

### 代码转换工具

```python
# src/data/code_mapper.py

def to_futu_code(ticker: str, market: str = 'US') -> str:
    """Convert standard ticker to Futu format"""
    return f"{market}.{ticker}"

def from_futu_code(futu_code: str) -> str:
    """Convert Futu code to standard ticker"""
    return futu_code.split('.', 1)[1]

def yfinance_to_futu(yf_ticker: str) -> str:
    """yfinance format to Futu format"""
    # yfinance: AAPL -> Futu: US.AAPL
    # yfinance: 0700.HK -> Futu: HK.00700
    if yf_ticker.endswith('.HK'):
        code = yf_ticker.replace('.HK', '').zfill(5)
        return f'HK.{code}'
    return f'US.{yf_ticker}'

def futu_to_yfinance(futu_code: str) -> str:
    """Futu format to yfinance format"""
    market, ticker = futu_code.split('.', 1)
    if market == 'HK':
        return f'{int(ticker)}.HK'
    return ticker
```

---

## 6. 佣金与费用结构

### 美股交易费用（富途证券）

#### 佣金
- **费率**: $0.0049/股
- **最低**: $0.99/笔
- **最高**: 成交金额的 0.5%

#### 平台使用费（阶梯式）

| 月交易量 | 每股费率 | 最低/笔 |
|----------|----------|---------|
| 1-500股 | $0.0100 | $1.00 |
| 501-1,000股 | $0.0080 | $1.00 |
| 1,001-5,000股 | $0.0070 | $1.00 |
| 5,001-10,000股 | $0.0060 | $1.00 |
| 10,000股+ | $0.0050 | $1.00 |

#### 监管费用（卖出时收取）
- **SEC Fee**: 0.00278% × 成交金额
- **TAF**: $0.000166/股，最低$0.01，最高$8.30
- **FINRA Fee**: 0.00145% × 成交金额

### 费用计算示例

```python
# 买入 100股 @ $50 = $5,000
def calculate_buy_cost(shares: int, price: float, tier: int = 3) -> dict:
    trade_value = shares * price
    
    # Commission
    commission = max(0.99, 0.0049 * shares)
    commission = min(commission, trade_value * 0.005)
    
    # Platform fee (tier 3 assumed)
    tier_rates = {1: 0.0100, 2: 0.0080, 3: 0.0070, 4: 0.0060, 5: 0.0050}
    platform_fee = max(1.00, tier_rates.get(tier, 0.0070) * shares)
    
    # FINRA fee (both sides)
    finra_fee = trade_value * 0.0000145
    
    total = commission + platform_fee + finra_fee
    
    return {
        'commission': commission,
        'platform_fee': platform_fee,
        'finra_fee': finra_fee,
        'total': total,
        'total_pct': total / trade_value * 100
    }

# Example: Buy 100 shares @ $50
# Commission: $0.99 (min applies)
# Platform: $1.00 (tier 3, min applies)
# FINRA: $0.07
# Total: ~$2.06 (0.04% of trade)
```

---

## 7. OpenD 网关配置

### 配置文件 (FutuOpenD.xml)

```xml
<?xml version="1.0" encoding="UTF-8"?>
<OpenD>
    <Api>
        <Ip>127.0.0.1</Ip>
        <Port>11111</Port>
    </Api>
    
    <Account>
        <!-- 牛牛号 -->
        <AccountId>YOUR_NN_ID</AccountId>
        <!-- 登录密码 MD5 -->
        <LoginPasswordMD5>YOUR_MD5_PASSWORD_MD5</LoginPasswordMD5>
    </Account>
    
    <Log>
        <Level>info</Level>
        <Path>./log</Path>
    </Log>
</OpenD>
```

### Docker Compose 配置

```yaml
version: '3.8'

services:
  opend:
    image: futu-opend:latest  # 需自行构建
    volumes:
      - ./opend/FutuOpenD.xml:/app/FutuOpenD.xml
      - ./opend/log:/app/log
    ports:
      - "11111:11111"
    restart: always
    # 注意：首次启动需要验证码，建议先在宿主机配置好

  quant-mvp:
    build: .
    depends_on:
      - opend
    environment:
      - FUTU_OPEND_HOST=opend
      - FUTU_OPEND_PORT=11111
      - FUTU_TRADE_PASSWORD_MD5=${FUTU_TRADE_PASSWORD_MD5}
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
```

### 进程监控脚本

```python
# src/execution/opend_monitor.py
"""Monitor and auto-restart FutuOpenD gateway"""

import subprocess
import time
import socket
from src.ops.event_logger import get_logger

logger = get_logger()

class OpenDMonitor:
    def __init__(self, host='127.0.0.1', port=11111, check_interval=30):
        self.host = host
        self.port = port
        self.check_interval = check_interval
        self.process = None
    
    def is_connected(self) -> bool:
        """Check if OpenD is reachable"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex((self.host, self.port))
            sock.close()
            return result == 0
        except Exception:
            return False
    
    def restart(self):
        """Restart OpenD process"""
        logger.error("opend_restart_initiated", {"host": self.host, "port": self.port})
        
        # Kill existing process
        if self.process:
            self.process.terminate()
            self.process.wait(timeout=10)
        
        # Start new process
        self.process = subprocess.Popen(
            ['./opend/FutuOpenD', '-c', './opend/FutuOpenD.xml'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Wait for startup
        time.sleep(5)
        
        if self.is_connected():
            logger.info("opend_restart_success")
            return True
        else:
            logger.error("opend_restart_failed")
            return False
    
    def run(self):
        """Main monitoring loop"""
        while True:
            if not self.is_connected():
                self.restart()
            time.sleep(self.check_interval)
```

---

## 8. 关键限制与注意事项

### 订阅额度限制

| 行情类型 | 免费额度 | 说明 |
|----------|----------|------|
| 实时报价 (QUOTE) | 100只 | 同时订阅的股票数 |
| 实时摆盘 (ORDER_BOOK) | 100只 | Level 2 数据 |
| 实时K线 | 100只 | 订阅后推送 |

**应对策略**：
- 仅在需要时订阅，用完立即取消
- 优先使用 yfinance 获取历史数据
- Futu 主要用于实时 bid/ask 和交易执行

### 交易限制

- **订单频率**: 默认 10笔/秒
- **查询频率**: 默认 20次/秒
- **解锁有效期**: 单次解锁后可持续交易，超时需重新解锁

### 模拟盘限制

- 美股模拟账户尚未支持融资融券模式
- 成交逻辑可能与真实市场有偏差
- **建议**: Phase E 缩短至 3-4 周，尽快进入 Phase F 实盘验证

---

## 9. 错误处理

### 常见错误码

```python
from futu import RET_OK

ret, data = trade_ctx.place_order(...)

if ret != RET_OK:
    # Handle error
    error_codes = {
        -1: "网络错误",
        -2: "超时",
        -3: "参数错误",
        -4: "未登录",
        -5: "未解锁交易",
        -100: "未知错误"
    }
    logger.error("order_failed", {"code": ret, "msg": data})
```

### 最佳实践

1. **始终检查返回值**: `if ret != RET_OK:`
2. **解锁交易**: 实盘下单前必须调用 `unlock_trade()`
3. **异常重试**: 网络错误时指数退避重试
4. **订单查询**: 下单后查询确认状态
5. **幂等性**: 使用 `remark` 字段存储 intent_id

---

## 10. 与项目集成要点

### 执行器封装

```python
# src/execution/futu_executor.py

class FutuExecutor:
    """Futu OpenAPI executor with idempotency support"""
    
    def __init__(self, host='127.0.0.1', port=11111, env='SIMULATE'):
        self.trd_ctx = OpenSecTradeContext(
            filter_trdmarket=TrdMarket.US,
            host=host, port=port
        )
        self.quote_ctx = OpenQuoteContext(host=host, port=port)
        self.env = TrdEnv.SIMULATE if env == 'SIMULATE' else TrdEnv.REAL
    
    def unlock(self, password_md5: str):
        """Unlock trading for live environment"""
        if self.env == TrdEnv.REAL:
            ret, data = self.trd_ctx.unlock_trade(password_md5=password_md5)
            if ret != RET_OK:
                raise RuntimeError(f"Unlock failed: {data}")
    
    def place_limit_order(self, ticker: str, side: str, 
                          qty: int, price: float,
                          intent_id: str,
                          extended_hours: bool = True) -> dict:
        """Place limit order with idempotency"""
        
        code = to_futu_code(ticker)
        trd_side = TrdSide.BUY if side == 'BUY' else TrdSide.SELL
        
        ret, data = self.trd_ctx.place_order(
            price=price,
            qty=qty,
            code=code,
            trd_side=trd_side,
            order_type=OrderType.NORMAL,
            trd_env=self.env,
            time_in_force=TimeInForce.DAY,
            fill_outside_rth=extended_hours,
            remark=f"intent:{intent_id}"[:64]  # 64 byte limit
        )
        
        if ret != RET_OK:
            raise OrderError(f"Place order failed: {data}")
        
        return {
            'order_id': str(data['order_id'][0]),
            'status': data['order_status'][0]
        }
    
    def get_mid_price(self, ticker: str) -> float:
        """Get mid price from order book"""
        code = to_futu_code(ticker)
        
        # Subscribe first
        self.quote_ctx.subscribe([code], [SubType.ORDER_BOOK])
        
        ret, data = self.quote_ctx.get_order_book(code)
        if ret != RET_OK:
            raise QuoteError(f"Get order book failed: {data}")
        
        bid = data['Bid'][0][0] if data['Bid'][0] else None
        ask = data['Ask'][0][0] if data['Ask'][0] else None
        
        if bid and ask:
            return (bid + ask) / 2
        raise QuoteError(f"No bid/ask for {ticker}")
```

### 环境变量配置

```bash
# .env
FUTU_OPEND_HOST=127.0.0.1
FUTU_OPEND_PORT=11111
FUTU_TRADE_PASSWORD_MD5=your_md5_hashed_password
FUTU_SECURITY_FIRM=FUTUINC  # FUTUINC for moomoo US, FUTUSECURITIES for HK
```

---

## 11. 参考链接

- [Futu OpenAPI 官方文档](https://openapi.futunn.com/futu-api-doc/)
- [Futu OpenAPI Python SDK](https://github.com/FutunnOpen/futu-api)
- [富途美股收费说明](https://www.futuhk.com/hans/support/topic2_283)
- [Moomoo US Pricing](https://www.moomoo.com/us/pricing)

---

*Last Updated: 2026-02-24*
