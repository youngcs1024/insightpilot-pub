---
title: 已完成退款笔数口径备忘录
doc_type: metric_memo
effective_from: 2025-03-01
effective_to: null
supersedes: null
metric_key: refund_count
---

# 已完成退款笔数口径备忘录

## 规范定义

指标：refund_count；版本：1；名称：已完成退款笔数

业务时区 Asia/Shanghai，期间为半开区间 [start,end)。合格订单已支付、未取消且非测试账号。区域采用下单区域 orders.region_id。按 requested_at 归入期间，统计最终状态 completed 的退款记录，按 refund_id 计数，不按订单去重；不按 completed_at 切期间。退款状态知识截至种子观察截止 2026-12-15T00:00:00+08:00。

默认日期字段：r.requested_at

必需过滤条件：
- o.paid_at IS NOT NULL
- o.status <> 'cancelled'
- c.is_test_account = false
- r.status = 'completed'

支持粒度：total、day、week、month、region

来源表：biz.orders、biz.customers、biz.refunds

## 使用说明

这项指标数的是退款记录，不是申请退款的订单。一张订单发生两笔最终完成的部分退款，可贡献两笔 refund_count，却只贡献一个申请退款订单。期间仍依据 requested_at；不要看到“已完成”就改成 completed_at。财务需要到账月份台账时须单独定义查询，不得复用本指标名掩盖时间口径改变。
