# Phase B 修复 + 技术债务清理计划

## Phase B 阻塞项 (来自 Audit v2)
- [ ] B14: VIX 变化率 + 市场宽度特征 (HIGH)
- [ ] B15: 量价背离特征 (MEDIUM)
- [ ] B20: 类别不平衡监控 (MEDIUM)
- [ ] B23: test_no_leakage.py 特征层测试补全 (MEDIUM)

## 观察项技术债务 (来自 Audit v4 FINAL)
- [ ] O1: integrity.py freeze_data() 使用 WAP (LOW)
- [ ] O2: daily_job.py JSON 写入使用 os.replace() (LOW)
- [ ] O3: ingest.py _create_source() 绑定 YAML 参数 (LOW)
- [ ] O4: universe.py Patch 5 字段 (rebalance_date_source + evidence) (LOW)
- [ ] O5: PDT 检查移至 risk/pdt_guard.py (LOW)

## 执行顺序
1. 先修 Phase B 阻塞项 (B14, B15, B20, B23)
2. 再清技术债务 (O1-O5)
