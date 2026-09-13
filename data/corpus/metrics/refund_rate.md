---
title: 退款申请订单率口径备忘录
doc_type: metric_memo
effective_from: 2025-03-01
effective_to: null
supersedes: null
metric_key: refund_rate
---

# 退款申请订单率口径备忘录

## 规范定义

指标：refund_rate；版本：1；名称：退款申请订单率

业务时区 Asia/Shanghai，期间为半开区间 [start,end)。合格订单已支付、未取消且非测试账号。区域采用下单区域 orders.region_id。分子按 requested_at 统计期间非 rejected 的退款申请去重订单数，分母按 paid_at 统计同期合格支付订单数。两侧独立聚合避免多笔退款放大；零分母返回 NULL。这是流量比值，可超过 100%，不是支付同期群概率。退款状态知识截至种子观察截止 2026-12-15T00:00:00+08:00；后续退款月份没有同期订单分母时不报告为 0。

默认日期字段：r.requested_at

必需过滤条件：
- o.paid_at IS NOT NULL
- o.status <> 'cancelled'
- c.is_test_account = false
- r.status <> 'rejected'

支持粒度：total、day、week、month、region

来源表：biz.orders、biz.customers、biz.refunds

## 使用说明

七月支付而八月申请的订单，会进入八月分子；八月支付而九月申请的订单，不进入八月申请期分子。两侧是不同的时间流量，比例超过 100% 并不自动说明 SQL 错误。T7 明确要求另一种支付同期群比较，应单独标明观察截止时间，不能偷偷替换本默认指标。缺少同期支付订单的后续月份不能显示为零退款率。
