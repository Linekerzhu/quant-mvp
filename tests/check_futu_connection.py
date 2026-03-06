import futu as ft
from src.execution.futu_executor import FutuExecutor
import time

def main():
    print("====================================")
    print("富途 OpenAPI 连接探针")
    print("====================================")
    
    # 建立对象，TrdEnv.SIMULATE 可以访问虚拟资产
    executor = FutuExecutor(host='127.0.0.1', port=11111, trd_env=ft.TrdEnv.SIMULATE)
    
    print("\n[>] 尝试连接到 127.0.0.1:11111")
    connected = executor.connect()
    if not connected:
        print("[!] 连接失败。请确认 11111 端口已在网关上开启监听。")
        return
        
    print("[✓] 连接网关成功。")
    
    print("\n[>] 尝试查询模拟盘总资产...")
    time.sleep(1) # wait buffer
    acc_value = executor.get_account_value()
    print(f"[✓] 当前账户虚拟资金 (USD): ${acc_value:,.2f}")
    
    print("\n[>] 尝试查询模拟盘当前持仓...")
    positions = executor.get_positions()
    if positions.empty:
        print("[✓] 当前无持仓。")
    else:
        print(f"[✓] 有 {len(positions)} 个持仓标的。")
        print(positions[['code', 'qty', 'cost_price', 'val', 'pl_ratio']])
        
    executor.close()
    print("====================================")
    print("测试完毕。")

if __name__ == "__main__":
    main()
